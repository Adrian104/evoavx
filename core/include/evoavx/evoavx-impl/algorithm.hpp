#pragma once
#include "global.hpp"
#include "island-dec.hpp"
#include "static-settings.hpp"
#include "triplet.hpp"

namespace evo
{
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
