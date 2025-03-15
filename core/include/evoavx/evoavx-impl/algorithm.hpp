#pragma once
#include "global.hpp"
#include "island-dec.hpp"

namespace evo::cc
{
	template <typename TripletT, typename S>
	concept triplet = requires(Island<S> island,
		typename TripletT::template selection_t<S> selection,
		typename TripletT::template crossover_t<S> crossover,
		typename TripletT::template mutation_t<S> mutation)
	{
		requires static_settings<S>;
		selection.perform_selection(island);
	};
}

namespace evo
{
	template <
		template <typename> typename SelectionT,
		template <typename> typename CrossoverT,
		template <typename> typename MutationT>
	class Triplet
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
	};

	template <cc::static_settings S, cc::triplet<S> TripletT>
	class Algorithm : public AlgorithmBase<S>
	{
	public:
		using selection_t = typename TripletT::template selection_t<S>;
		using crossover_t = typename TripletT::template crossover_t<S>;
		using mutation_t = typename TripletT::template mutation_t<S>;

		selection_t m_selection;
		crossover_t m_crossover;
		mutation_t m_mutation;
	};
}
