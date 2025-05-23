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

	class Link
	{
	public:
		int m_neighbor = 0;
		u64_t m_counter = 0;
		u64_t m_migrants = 0;
		unique<f64_t[]> m_send;
		unique<f64_t[]> m_recv;
	};

	template <cc::static_settings S>
	class Island
	{
	public:
		std::vector<Link> m_links;
		std::vector<std::pair<f64_t, f64_t>> m_genome;

		Component<AlgorithmBase<S>> m_algorithm;
		Component<EvaluatorBase<S>> m_evaluator;
		Extremum m_extremum = Extremum::MINIMUM;
		Statistics<S> m_statistics;
		Random<S> m_random;

		unique<f64_t[]> m_current;
		unique<f64_t[]> m_next;
		unique<f64_t[]> m_scores;
		unique<u64_t[]> m_selected;
		unique<f64_t[]> m_minDomain;
		unique<f64_t[]> m_maxDomain;

		MPI_Comm m_communicator{};
		int m_rank = 0;

		u32_t m_cacheExponent = 0;
		u64_t m_indivCount = 0;
		u64_t m_selIndivCount = 0;
		u64_t m_migIndivCount = 0;
		u64_t m_migInterval = 0;
		u64_t m_realGenomeLength = 0;
		u64_t m_seed = 0;

		f64_t m_crossoverProb = 0;
		f64_t m_mutationProb = 0;

		Island() = default;
		~Island() = default;

		Island(const Island<S>&) = delete;
		Island& operator=(const Island<S>&) = delete;

		Island(Island<S>&&) = delete;
		Island& operator=(Island<S>&&) = delete;

		void init();
		void entry_point();
	};
}
