#pragma once
#include "global.hpp"
#include "algorithm.hpp"
#include "evaluator.hpp"
#include "island-dec.hpp"
#include "statistics.hpp"

namespace evo
{
	template <cc::static_settings S>
	inline Island<S>::Island(Shared<S>& shared, u64_t id)
		: m_shared(shared), m_islandId(id) {}

	template <cc::static_settings S>
	inline void Island<S>::init()
	{
		m_statistics.start();
		m_random.init(m_shared.m_seed);

		for (u64_t i = 0; i < m_islandId; i++)
			m_random.jump();

		m_genomeLength = m_shared.m_genome.size();
		m_realGenomeLength = alignment_ceil<f64_t>(m_genomeLength);

		u64_t genesTotal = (m_indivCount + (m_indivCount & 1)) * m_realGenomeLength;
		u64_t scoresTotal = alignment_ceil<f64_t>(m_indivCount);
		u64_t selectedTotal = alignment_ceil<u64_t>(m_selIndivCount);

		m_current = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		m_next = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		m_scores = unique<f64_t[]>(allocate<f64_t>(scoresTotal));
		m_selected = unique<u64_t[]>(allocate<u64_t>(selectedTotal));
		m_minDomain = unique<f64_t[]>(allocate<f64_t>(m_realGenomeLength));
		m_maxDomain = unique<f64_t[]>(allocate<f64_t>(m_realGenomeLength));

		const std::pair<f64_t, f64_t>* genome = m_shared.m_genome.data();
		for (u64_t i = 0, j = 0; i < m_realGenomeLength; ++i, ++j)
		{
			if (j == m_genomeLength)
				j = 0;

			m_minDomain[i] = genome[j].first;
			m_maxDomain[i] = genome[j].second;
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
	}

	template <cc::static_settings S>
	inline void Island<S>::entry_point()
	{
		init();
	}
}
