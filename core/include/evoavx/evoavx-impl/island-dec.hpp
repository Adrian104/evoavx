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
	class Shared
	{
	public:
		u64_t m_seed = 0;
		std::vector<std::pair<f64_t, f64_t>> m_genome;
	};

	template <cc::static_settings S>
	class Island
	{
	public:
		Component<AlgorithmBase<S>> m_algorithm;
		Component<EvaluatorBase<S>> m_evaluator;
		Extremum m_extremum = Extremum::MINIMUM;
		Statistics<S> m_statistics;
		Shared<S>& m_shared;
		Random<S> m_random;

		unique<f64_t[]> m_current;
		unique<f64_t[]> m_next;
		unique<f64_t[]> m_scores;
		unique<u64_t[]> m_selected;
		unique<f64_t[]> m_minDomain;
		unique<f64_t[]> m_maxDomain;

		const u64_t m_islandId;

		u64_t m_indivCount = 0;
		u64_t m_selIndivCount = 0;
		u64_t m_genomeLength = 0;
		u64_t m_realGenomeLength = 0;

		f64_t m_crossoverProb = 0;
		f64_t m_mutationProb = 0;

		Island(Shared<S>& shared, u64_t id);
		~Island() = default;

		Island(const Island<S>&) = delete;
		Island& operator=(const Island<S>&) = delete;

		Island(Island<S>&&) = delete;
		Island& operator=(Island<S>&&) = delete;

		void init();
		void entry_point();
	};
}
