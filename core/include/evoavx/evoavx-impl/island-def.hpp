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
		if constexpr (S::alignment_v == Alignment::ALIGNED)
			m_realGenomeLength = alignment_ceil<f64_t>(m_genomeLength);
		else
			m_realGenomeLength = m_genomeLength;

		u64_t genesTotal = alignment_ceil<f64_t>(m_indivCount * m_realGenomeLength);
		u64_t selectedTotal = alignment_ceil<f64_t>(m_selIndivCount * m_realGenomeLength);
		u64_t scoresTotal = alignment_ceil<f64_t>(m_indivCount);

		m_genes = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		m_selected = unique<f64_t[]>(allocate<f64_t>(selectedTotal));
		m_scores = unique<f64_t[]>(allocate<f64_t>(scoresTotal));
		m_scoresAux = unique<f64_t[]>(allocate<f64_t>(scoresTotal));

		u64_t domainTotal;
		if constexpr (S::alignment_v == Alignment::ALIGNED)
			domainTotal = m_realGenomeLength;
		else
			domainTotal = alignment_ceil<f64_t>(m_genomeLength + g_vectorGenes - 1);

		m_minDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));
		m_maxDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));
		m_diffDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));

		const std::pair<f64_t, f64_t>* genome = m_shared.m_genome.data();
		for (u64_t i = 0, j = 0; i < domainTotal; ++i, ++j)
		{
			if (j == m_genomeLength)
				j = 0;

			const auto& [a, b] = genome[j];

			m_minDomain[i] = a;
			m_maxDomain[i] = b;
			m_diffDomain[i] = b - a;
		}

		f64_t* const genesPtr = m_genes.get();
		const f64_t* const minPtr = m_minDomain.get();
		const f64_t* const diffPtr = m_diffDomain.get();

		if constexpr (S::alignment_v == Alignment::ALIGNED)
		{
			for (u64_t i = 0, j = 0; i < genesTotal; i += g_vectorGenes, j += g_vectorGenes)
			{
				if (j == m_realGenomeLength)
					j = 0;

				__m512d random = m_random.next_512d();
				__m512d min = _mm512_load_pd(minPtr + j);
				__m512d diff = _mm512_load_pd(diffPtr + j);
				__m512d values = _mm512_fmadd_pd(diff, random, min);

				_mm512_store_pd(genesPtr + i, values);
			}
		}
		else
		{
			for (u64_t i = 0; i < genesTotal; i += g_vectorGenes)
			{
				u64_t j = i % m_realGenomeLength;

				__m512d random = m_random.next_512d();
				__m512d min = _mm512_loadu_pd(minPtr + j);
				__m512d diff = _mm512_loadu_pd(diffPtr + j);
				__m512d values = _mm512_fmadd_pd(diff, random, min);

				_mm512_store_pd(genesPtr + i, values);
			}
		}
	}

	template <cc::static_settings S>
	inline void Island<S>::entry_point()
	{
		init();
	}
}
