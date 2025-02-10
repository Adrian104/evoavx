#pragma once
#include "global.hpp"
#include "state.hpp"

namespace evo::cc
{
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
}

namespace evo
{
	template <cc::static_settings S>
	class Selection
	{
	public:
		virtual ~Selection() = default;
	};

	template <cc::static_settings S>
	class FusedXMBase
	{
	public:
		virtual ~FusedXMBase() = default;
	};

	template <cc::static_settings S, cc::crossover CrossoverT, cc::mutation MutationT>
	class FusedXM : public FusedXMBase<S>, public CrossoverT, public MutationT
	{

	};
}
