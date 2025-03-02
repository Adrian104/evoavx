#pragma once
#include "global.hpp"
#include "component.hpp"
#include "random/random.hpp"

namespace evo
{
	template <cc::static_settings S>
	class AlgorithmBase;

	template <cc::static_settings S>
	class EvaluatorBase;

	template <cc::static_settings S>
	class Statistics;

	template <cc::static_settings S>
	class Island
	{
	public:
		Component<AlgorithmBase<S>> m_algorithm;
		Component<EvaluatorBase<S>> m_evaluator;
		Extremum m_extremum = Extremum::MINIMUM;
		Statistics<S> m_statistics;
		Random<S> m_random;

		unique<f64_t[]> m_genes;
		unique<f64_t[]> m_selected;
		unique<f64_t[]> m_scores;
		unique<f64_t[]> m_scoresAux;
		unique<f64_t[]> m_minDomain;
		unique<f64_t[]> m_maxDomain;
		unique<f64_t[]> m_diffDomain;

		u64_t m_indivCount = 0;
		u64_t m_selIndivCount = 0;
		u64_t m_genomeLength = 0;
		u64_t m_realGenomeLength = 0;

		f64_t m_crossoverProb = 0;
		f64_t m_mutationProb = 0;

		Island() = default;
		~Island() = default;

		Island(const Island<S>&) = delete;
		Island& operator=(const Island<S>&) = delete;

		Island(Island<S>&&) = delete;
		Island& operator=(Island<S>&&) = delete;
	};
}
