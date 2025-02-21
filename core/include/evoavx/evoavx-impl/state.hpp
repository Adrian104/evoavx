#pragma once
#include "global.hpp"
#include "component.hpp"
#include "static-settings.hpp"
#include "random/random.hpp"

namespace evo
{
	template <cc::static_settings S>
	class EvaluatorBase;

	enum class Extremum
	{
		MINIMUM,
		MAXIMUM
	};

	template <cc::static_settings S>
	class State
	{
	public:
		Random<typename S::prng_engine_t> m_random;
		Component<EvaluatorBase<S>> m_evaluator;
		std::vector<std::pair<f64_t, f64_t>> m_genome;
		Extremum m_extremum = Extremum::MINIMUM;

		unique<f64_t[]> m_genes;
		unique<f64_t[]> m_selected;
		unique<f64_t[]> m_scores;
		unique<f64_t[]> m_minDomain;
		unique<f64_t[]> m_maxDomain;
		unique<f64_t[]> m_diffDomain;

		u64_t m_indivCount = 0;
		u64_t m_geneCount = 0;
		u64_t m_selIndivCount = 0;
		u64_t m_selGeneCount = 0;

		f64_t m_crossoverProb = 0;
		f64_t m_mutationProb = 0;

		void verify();
		void init();
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline void State<S>::verify()
	{
		if (m_evaluator.get_used() == nullptr)
			throw std::runtime_error("Fitness function is not set");

		if (m_genome.empty())
			throw std::runtime_error("Genome is empty");

		if (m_indivCount == 0 || m_selIndivCount == 0)
			throw std::runtime_error("Population size is not set");

		if (m_evaluator.get_used()->get_length() != m_genome.size())
			throw std::runtime_error("Genome length does not match the number of arguments required by the fitness function");
	}

	template <cc::static_settings S>
	inline void State<S>::init()
	{
		m_geneCount = m_indivCount * m_genome.size();
		m_selGeneCount = m_selIndivCount * m_genome.size();

		u64_t genesTotal = alignment_ceil<f64_t>(m_geneCount);
		u64_t selectedTotal = alignment_ceil<f64_t>(m_selGeneCount);
		u64_t scoresTotal = alignment_ceil<f64_t>(m_indivCount);

		m_genes = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		m_selected = unique<f64_t[]>(allocate<f64_t>(selectedTotal));
		m_scores = unique<f64_t[]>(allocate<f64_t>(scoresTotal));

		u64_t domainArraySize = m_genome.size() + g_vectorGenes - 1;
		u64_t domainTotal = alignment_ceil<f64_t>(domainArraySize);

		m_minDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));
		m_maxDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));
		m_diffDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));

		std::random_device rd;
		m_random.init(rd());

		for (u64_t i = 0, j = 0; i < domainArraySize; i++, j++)
		{
			if (j >= m_genome.size())
				j -= m_genome.size();

			auto& [a, b] = m_genome[j];

			m_minDomain[i] = a;
			m_maxDomain[i] = b;
			m_diffDomain[i] = b - a;
		}

		u64_t jcrr = 0;
		u64_t jnext = 0;

		for (u64_t i = 0; i < genesTotal; i += g_vectorGenes)
		{
			jcrr = jnext;
			jnext = (i + g_vectorGenes) % m_genome.size();

			__m512d random = m_random.next_512d();
			__m512d min = _mm512_loadu_pd(m_minDomain.get() + jcrr);
			__m512d diff = _mm512_loadu_pd(m_diffDomain.get() + jcrr);
			__m512d initValues = _mm512_fmadd_pd(diff, random, min);

			_mm512_store_pd(m_genes.get() + i, initValues);
		}
	}
}
