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
		requires cc::wide_prng_engine<typename S::prng_engine_t>;
		{ S::range_alg_v } -> std::convertible_to<RangeAlg>;
	};
}

namespace evo
{
	template <cc::wide_prng_engine PrngEngineT, RangeAlg rangeAlg>
	class StaticSettings
	{
	public:
		using prng_engine_t = PrngEngineT;
		constexpr static RangeAlg range_alg_v = rangeAlg;
	};

	using DefaultStaticSettings = StaticSettings<Xoshiro256pp<SplitMix64>, RangeAlg::LEMIRE_64>;
	static_assert(cc::static_settings<DefaultStaticSettings>);
}
