#pragma once
#include "global.hpp"
#include "island-dec.hpp"
#include "random/random.hpp"

namespace evo::cc
{
	template <typename B, typename S>
	concept blueprint = requires(Island<S> island, f64_t f64, __m512d m512d, Random<S> rand,
		typename B::template selection_t<S> selection,
		typename B::template crossover_t<S> crossover,
		typename B::template mutation_t<S> mutation)
	{
		requires static_settings<S>;

		{ decltype(selection)::s_usesAux } -> std::convertible_to<bool>;
		{ decltype(crossover)::s_twins } -> std::convertible_to<bool>;
		{ decltype(crossover)::s_fusedXM } -> std::convertible_to<bool>;
		{ decltype(crossover)::s_forceDomain } -> std::convertible_to<bool>;
		{ decltype(mutation)::s_fusedXM } -> std::convertible_to<bool>;

		selection.perform(island);
		crossover.template perform<decltype(mutation)>(island, &f64, &f64, &f64);
		mutation.template perform<decltype(crossover)::s_forceDomain>(island, &f64);

		{ decltype(mutation)::template perform<decltype(crossover)::s_forceDomain>
			(m512d, m512d, m512d, f64, rand) } -> std::same_as<__m512d>;

		crossover.init_generation(island);
		crossover.init_wave(island);
		mutation.init_generation(island);
		mutation.init_wave(island);
	};
}

namespace evo
{
	template <
		template <typename> typename SelectionT,
		template <typename> typename CrossoverT,
		template <typename> typename MutationT>
	class Blueprint
	{
	public:
		template <cc::static_settings S>
		using selection_t = SelectionT<S>;

		template <cc::static_settings S>
		using crossover_t = CrossoverT<S>;

		template <cc::static_settings S>
		using mutation_t = MutationT<S>;
	};

	template <cc::static_settings S>
	class AlgorithmBase
	{
	public:
		virtual ~AlgorithmBase() = default;
		virtual void phase_1(Island<S>& island) = 0;
		virtual void phase_2(Island<S>& island) = 0;
	};

	template <cc::static_settings S, cc::blueprint<S> B>
	class Algorithm : public AlgorithmBase<S>
	{
	public:
		using selection_t = typename B::template selection_t<S>;
		using crossover_t = typename B::template crossover_t<S>;
		using mutation_t = typename B::template mutation_t<S>;

		constexpr static bool s_twins = crossover_t::s_twins;
		constexpr static bool s_fusedXM = crossover_t::s_fusedXM && mutation_t::s_fusedXM && S::fused_xm_v == FusedXM::AUTO;
		constexpr static bool s_forceDomain = crossover_t::s_forceDomain || S::force_domain_v == ForceDomain::ENABLED;

		selection_t m_selection;
		crossover_t m_crossover;
		mutation_t m_mutation;

		void phase_1(Island<S>& island) override;
		void phase_2(Island<S>& island) override;
	};
}

namespace evo
{
	template <cc::static_settings S, cc::blueprint<S> B>
	inline void Algorithm<S, B>::phase_1(Island<S>& island)
	{
		EvaluatorBase<S>* const evaluator = island.m_evaluator.get_used();
		f64_t meanEstimate;

		if constexpr (selection_t::s_usesAux)
			meanEstimate = evaluator->evaluate_population_with_aux(island);
		else
			meanEstimate = evaluator->evaluate_population(island);

		island.m_statistics.update(island, meanEstimate);
	}

	template <cc::static_settings S, cc::blueprint<S> B>
	inline void Algorithm<S, B>::phase_2(Island<S>& island)
	{
		m_selection.perform(island);
	}
}
