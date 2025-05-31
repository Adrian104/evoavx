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

	}

	template <cc::static_settings S>
	inline void Island<S>::communicate()
	{

	}

	template <cc::static_settings S>
	inline void Island<S>::entry_point()
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
	}
}
