#pragma once
#include "global.hpp"
#include "component.hpp"
#include "random/random.hpp"

namespace evo
{
	class Inspector;

	template <cc::static_settings S>
	class AlgorithmBase;

	template <cc::static_settings S>
	class EvaluatorBase;

	template <cc::static_settings S>
	class StatisticsEngine;

	class Link
	{
	public:
		int m_neighbor = 0;
		u64_t m_counter = 0;
		u64_t m_migrants = 0;
		unique<f64_t[]> m_send;
		unique<f64_t[]> m_recv;
	};

	class Result
	{
	public:
		f64_t m_wtime = 0;
		f64_t m_score = 0;
		std::vector<f64_t> m_args;
	};

	template <cc::static_settings S>
	class Island
	{
	public:
		enum class State { RUNNING, FINALIZING, IDLE };
		State m_state = State::IDLE;

		std::vector<Link> m_links;
		std::vector<MPI_Request> m_requests;
		std::vector<std::pair<f64_t, f64_t>> m_genome;

		Component<Inspector> m_inspector;
		Component<AlgorithmBase<S>> m_algorithm;
		Component<EvaluatorBase<S>> m_evaluator;

		Extremum m_extremum = Extremum::MINIMUM;
		StatisticsEngine<S> m_statistics;
		Random<S> m_random;

		unique<f64_t[]> m_current;
		unique<f64_t[]> m_next;
		unique<f64_t[]> m_scores;
		unique<u64_t[]> m_selected;
		unique<f64_t[]> m_minDomain;
		unique<f64_t[]> m_maxDomain;

		MPI_Comm m_communicator{};
		MPI_Request m_shutdownRequest{};
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
		void check_stop_condition();
		void communicate();
		Result reduce();
		Result entry_point();
	};
}
