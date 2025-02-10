#pragma once
#include "global.hpp"
#include "component.hpp"
#include "fitness-function.hpp"
#include "static-settings.hpp"
#include "random/random.hpp"

namespace evo
{
	enum class Extremum
	{
		MINIMUM,
		MAXIMUM
	};

	template <cc::static_settings S>
	class State
	{
	public:
		std::unique_ptr<f64_t> m_genes;
		std::unique_ptr<f64_t> m_selected;
		std::unique_ptr<f64_t> m_scores;
		std::unique_ptr<f64_t> m_minDomain;
		std::unique_ptr<f64_t> m_maxDomain;
		std::unique_ptr<f64_t> m_diffDomain;

		Random<typename S::prng_engine_t> m_random;
		Component<FitnessFunction> m_fitnessFunc;
		std::vector<std::pair<f64_t, f64_t>> m_genome;
		Extremum m_extremum = Extremum::MINIMUM;

		u64_t m_individualCount = 0;
		u64_t m_geneCount = 0;
		u64_t m_selectedIndividualCount = 0;
		u64_t m_selectedGeneCount = 0;

		f64_t m_crossoverProb = 0.9;
		f64_t m_mutationProb = 0.2;
	};
}
