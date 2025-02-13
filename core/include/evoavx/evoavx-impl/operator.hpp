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

	template <cc::static_settings S, template <typename> typename X, template <typename> typename M>
		requires cc::crossover<X<S>> && cc::mutation<M<S>>
	class FusedXM : public FusedXMBase<S>, public X<S>, public M<S>
	{
	public:
		virtual ~FusedXM() = default;
	};
}
