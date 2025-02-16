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
	};
}
