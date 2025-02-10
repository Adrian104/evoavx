#pragma once
#include "global.hpp"
#include "random/random.hpp"
#include "random/splitmix64.hpp"
#include "random/xoshiro256pp.hpp"

namespace evo::cc
{
	template <typename S>
	concept static_settings = requires
	{
		typename S::prng_engine_t;
	};
}

namespace evo
{
	template <cc::wide_prng_engine PrngEngineT>
	class StaticSettings
	{
	public:
		using prng_engine_t = PrngEngineT;
	};

	using DefaultStaticSettings = StaticSettings<Xoshiro256pp<SplitMix64>>;
	static_assert(cc::static_settings<DefaultStaticSettings>);
}
