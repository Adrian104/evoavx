#pragma once
#include "global.hpp"
#include "algorithm.hpp"
#include "evaluator.hpp"
#include "island-dec.hpp"
#include "shuffle.hpp"
#include "statistics.hpp"

namespace evo
{
	template <cc::static_settings S>
	inline void Island<S>::init()
	{
		m_state = State::RUNNING;
		m_statistics.start();
		m_random.init(m_seed);
		m_inspector.get_used()->init();

		MPI_Comm_rank(m_communicator, &m_rank);
		for (int i = 0; i < m_rank; i++)
			m_random.jump();

		m_realGenomeLength = alignment_ceil<f64_t>(m_genome.size());

		u64_t genesTotal = (m_indivCount + (m_indivCount & 1)) * m_realGenomeLength;
		u64_t scoresTotal = alignment_ceil<f64_t>(m_indivCount);
		u64_t selectedTotal = alignment_ceil<u64_t>(std::max(m_indivCount - 1, m_selIndivCount));

		m_current = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		m_next = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		m_scores = unique<f64_t[]>(allocate<f64_t>(scoresTotal));
		m_selected = unique<u64_t[]>(allocate<u64_t>(selectedTotal));
		m_minDomain = unique<f64_t[]>(allocate<f64_t>(m_realGenomeLength));
		m_maxDomain = unique<f64_t[]>(allocate<f64_t>(m_realGenomeLength));

		for (u64_t i = 0, j = 0; i < m_realGenomeLength; ++i, ++j)
		{
			if (j == m_genome.size())
				j = 0;

			m_minDomain[i] = m_genome[j].first;
			m_maxDomain[i] = m_genome[j].second;
		}

		f64_t* const genesPtr = m_current.get();
		const f64_t* const minPtr = m_minDomain.get();
		const f64_t* const maxPtr = m_maxDomain.get();

		for (u64_t i = 0, j = 0; i < genesTotal; i += g_vectorGenes, j += g_vectorGenes)
		{
			if (j == m_realGenomeLength)
				j = 0;

			__m512d min = _mm512_load_pd(minPtr + j);
			__m512d max = _mm512_load_pd(maxPtr + j);
			__m512d diff = _mm512_sub_pd(max, min);
			__m512d random = m_random.next_512d();
			__m512d values = _mm512_fmadd_pd(diff, random, min);

			_mm512_store_pd(genesPtr + i, values);
		}

		m_evaluator.get_used()->init_cache(m_cacheExponent, m_genome.size());

		int count;
		MPI_Graph_neighbors_count(m_communicator, m_rank, &count);

		if (count <= 0)
			return;

		m_requests.resize(count << 1);
		m_links.resize(count);

		std::vector<int> neighbors(count);
		std::vector<u64_t> migrations(count << 1, m_migIndivCount);

		MPI_Graph_neighbors(m_communicator, m_rank, neighbors.size(), neighbors.data());
		MPI_Neighbor_alltoall(migrations.data() + count, 1, MPI_UINT64_T, migrations.data(), 1, MPI_UINT64_T, m_communicator);

		for (int i = 0; i < count; i++)
		{
			Link& crr = m_links[i];

			const u64_t migrants = std::min(migrations[i], m_migIndivCount);
			const u64_t capacity = alignment_ceil<f64_t>(migrants * m_genome.size());

			crr.m_neighbor = neighbors[i];
			crr.m_migrants = migrants;
			crr.m_send = unique<f64_t[]>(allocate<f64_t>(capacity));
			crr.m_recv = unique<f64_t[]>(allocate<f64_t>(capacity));
		}
	}

	template <cc::static_settings S>
	inline void Island<S>::check_stop_condition()
	{
		int flag;
		const Action action = m_inspector.get_used()->inspect(m_statistics);

		switch (m_state)
		{
		case State::RUNNING:
			if (action != Action::STOP)
				break;

			MPI_Ibarrier(m_communicator, &m_shutdownRequest);
			[[fallthrough]];

		case State::FINALIZING:
			MPI_Test(&m_shutdownRequest, &flag, MPI_STATUS_IGNORE);
			m_state = flag ? State::IDLE : State::FINALIZING;
			[[fallthrough]];

		default:
			break;
		}
	}

	template <cc::static_settings S>
	inline void Island<S>::communicate()
	{
		int flag;
		MPI_Iprobe(MPI_ANY_SOURCE, g_migrationTag, m_communicator, &flag, MPI_STATUS_IGNORE);

		if (m_state == State::RUNNING)
		{
			for (Link& link : m_links)
				flag |= ++link.m_counter >= m_migInterval;
		}

		if (!flag)
			return;

		u64_t* const sel = m_selected.get();
		const u64_t selCount = m_indivCount - 1;
		const u64_t bestIdx = m_extremum == Extremum::MAXIMUM ? m_statistics.m_maximumPos : m_statistics.m_minimumPos;

		__m512i val = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
		__m512i inc1 = _mm512_set1_epi64(1);
		__m512i inc8 = _mm512_set1_epi64(8);
		__m512i excl = _mm512_set1_epi64(bestIdx);

		for (u64_t i = 0; i < selCount; i += g_vectorGenes)
		{
			__mmask8 cmp = _mm512_cmple_epu64_mask(excl, val);
			__m512i out = _mm512_mask_add_epi64(val, cmp, val, inc1);

			_mm512_store_epi64(sel + i, out);
			val = _mm512_add_epi64(val, inc8);
		}

		shuffle(sel, selCount, m_random);

		u64_t index = 0;
		MPI_Request* req = m_requests.data();
		f64_t* const local = m_current.get();
		const std::size_t bytes = m_genome.size() * sizeof(f64_t);

		for (Link& link : m_links)
		{
			MPI_Iprobe(link.m_neighbor, g_migrationTag, m_communicator, &flag, MPI_STATUS_IGNORE);
			if (m_state == State::RUNNING)
				flag |= link.m_counter >= m_migInterval;

			if (!flag)
				continue;

			f64_t* remote = link.m_send.get();
			std::memcpy(remote, local + bestIdx * m_realGenomeLength, bytes);

			for (u64_t i = 1; i < link.m_migrants; i++)
			{
				std::memcpy(remote += m_genome.size(), local + sel[index++] * m_realGenomeLength, bytes);
				if (index >= selCount)
					index = 0;
			}

			link.m_counter = std::numeric_limits<u64_t>::max();
			const int cnt = static_cast<int>(link.m_migrants * m_genome.size());

			MPI_Isend(link.m_send.get(), cnt, MPI_DOUBLE, link.m_neighbor, g_migrationTag, m_communicator, req++);
			MPI_Irecv(link.m_recv.get(), cnt, MPI_DOUBLE, link.m_neighbor, g_migrationTag, m_communicator, req++);
		}

		MPI_Waitall(req - m_requests.data(), m_requests.data(), MPI_STATUSES_IGNORE);

		u64_t total = 0;
		index = 0;

		for (Link& link : m_links)
		{
			if (link.m_counter != std::numeric_limits<u64_t>::max())
				continue;

			link.m_counter = 0;
			total += link.m_migrants;
			const f64_t* remote = link.m_recv.get();

			for (u64_t i = 0; i < link.m_migrants; i++)
			{
				std::memcpy(local + sel[index++] * m_realGenomeLength, remote, bytes);
				remote += m_genome.size();

				if (index >= selCount)
					index = 0;
			}
		}

		m_evaluator.get_used()->evaluate_subset(*this, std::min(total, selCount));
		m_statistics.refresh(*this);
	}

	template <cc::static_settings S>
	inline Result Island<S>::reduce()
	{
		struct
		{
			double m_score;
			int m_rank;

		} local, global;

		const auto [localBestScore, localBestPos, op] = m_extremum == Extremum::MAXIMUM
			? std::make_tuple(m_statistics.m_maximum, m_statistics.m_maximumPos, MPI_MAXLOC)
			: std::make_tuple(m_statistics.m_minimum, m_statistics.m_minimumPos, MPI_MINLOC);

		local.m_score = localBestScore;
		local.m_rank = m_rank;

		MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, op, m_communicator);

		Result result;
		result.m_score = global.m_score;
		result.m_args.resize(m_genome.size());

		f64_t* const buffer = result.m_args.data();
		const f64_t* const src = m_current.get() + localBestPos * m_realGenomeLength;

		std::memcpy(buffer, src, m_genome.size() * sizeof(f64_t));
		MPI_Bcast(buffer, m_genome.size(), MPI_DOUBLE, global.m_rank, m_communicator);

		m_inspector.get_used()->finish(result);
		return result;
	}

	template <cc::static_settings S>
	inline Result Island<S>::entry_point()
	{
		init();
		while (true)
		{
			const f64_t meanEstimate = m_evaluator.get_used()->evaluate_population(*this);
			m_statistics.update(*this, meanEstimate);
			check_stop_condition();

			if (m_state == State::IDLE)
				break;

			communicate();
			m_algorithm.get_used()->evolve(*this);
		}

		return reduce();
	}
}
