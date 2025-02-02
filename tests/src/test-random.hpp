#include "pch.hpp"
#include <evoavx/evoavx.hpp>
#include "reference/splitmix64.hpp"
#include "reference/xoshiro256pp.hpp"

TEST_CASE("Random-related types have correct properties")
{
	STATIC_REQUIRE(evo::cc::basic_prng_engine<evo::SplitMix64>);
	STATIC_REQUIRE(evo::cc::wide_prng_engine<evo::Xoshiro256pp<evo::SplitMix64>>);
}

TEST_CASE("Xoshiro256pp<SplitMix64> works as expected")
{
	constexpr static evo::u64_t s_seed = 42;
	constexpr static int s_jumps = 3;
	constexpr static int s_iterations = 10;

	std::vector<std::array<evo::u64_t, 8>> reference(s_iterations);

	for (int j = 0; j < 8; j++)
	{
		splitmix64::x = s_seed;
		xoshiro256pp::s[0] = splitmix64::next();
		xoshiro256pp::s[1] = splitmix64::next();
		xoshiro256pp::s[2] = splitmix64::next();
		xoshiro256pp::s[3] = splitmix64::next();

		for (int i = 0; i < s_jumps; i++)
			xoshiro256pp::jump();

		for (int i = 0; i < j; i++)
			xoshiro256pp::long_jump();

		for (int i = 0; i < s_iterations; i++)
			reference[i][j] = xoshiro256pp::next();
	}

	evo::Xoshiro256pp<evo::SplitMix64> prng;
	prng.init(s_seed);

	for (int i = 0; i < s_jumps; i++)
		prng.jump();

	for (int i = 0; i < s_iterations; i++)
	{
		std::array<evo::u64_t, 8> crr{};
		_mm512_storeu_epi64(crr.data(), prng.next());
		REQUIRE(crr == reference[i]);
	}
}
