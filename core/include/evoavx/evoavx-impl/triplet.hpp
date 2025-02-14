#pragma once
#include "global.hpp"

namespace evo::cc
{
	template <typename SelectionT>
	concept selection = requires
	{
		1;
	};

	template <typename CrossoverT>
	concept crossover = requires
	{
		1;
	};

	template <typename MutationT>
	concept mutation = requires
	{
		1;
	};

	template <typename TripletT, typename S>
	concept triplet = requires
	{
		requires static_settings<S>;
		requires selection<typename TripletT::template selection_t<S>>;
		requires crossover<typename TripletT::template crossover_t<S>>;
		requires mutation<typename TripletT::template mutation_t<S>>;
	};
}

namespace evo
{
	template <
		template <typename> typename SelectionT,
		template <typename> typename CrossoverT,
		template <typename> typename MutationT>
	requires
		cc::selection<SelectionT<DefaultStaticSettings>> &&
		cc::crossover<CrossoverT<DefaultStaticSettings>> &&
		cc::mutation<MutationT<DefaultStaticSettings>>
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
}
