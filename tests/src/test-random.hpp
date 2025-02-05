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

	evo::Xoshiro256pp<evo::SplitMix64> prng(s_seed);

	for (int i = 0; i < s_jumps; i++)
		prng.jump();

	for (int i = 0; i < s_iterations; i++)
	{
		std::array<evo::u64_t, 8> crr{};
		_mm512_storeu_epi64(crr.data(), prng.next_512i());
		REQUIRE(crr == reference[i]);
	}
}

TEST_CASE("Method next_512d() works as expected")
{
	constexpr static evo::u64_t s_seed = 42;
	constexpr static int s_iterations = 1 << 12;

	evo::Random<evo::Xoshiro256pp<evo::SplitMix64>> prng(s_seed);

	__m512d sum = _mm512_setzero_pd();
	__m512d max = _mm512_set1_pd(-10.0);
	__m512d min = _mm512_set1_pd(10.0);

	for (int i = 0; i < s_iterations; i++)
	{
		__m512d crr = prng.next_512d();
		sum = _mm512_add_pd(sum, crr);
		max = _mm512_max_pd(max, crr);
		min = _mm512_min_pd(min, crr);
	}

	std::array<evo::f64_t, 8> sumArr{};
	std::array<evo::f64_t, 8> maxArr{};
	std::array<evo::f64_t, 8> minArr{};

	_mm512_storeu_pd(sumArr.data(), sum);
	_mm512_storeu_pd(maxArr.data(), max);
	_mm512_storeu_pd(minArr.data(), min);

	for (auto v : sumArr)
		REQUIRE_THAT(v / s_iterations, Catch::Matchers::WithinAbs(0.5, 0.1));

	for (auto v : maxArr)
	{
		REQUIRE(v <= 1.0);
		REQUIRE(v > 0.9);
	}

	for (auto v : minArr)
	{
		REQUIRE(v >= 0.0);
		REQUIRE(v < 0.1);
	}
}

TEST_CASE("Method range_512i() works as expected")
{
	constexpr static evo::u64_t s_seed = 42;
	constexpr static int s_iterations = 1 << 12;

	evo::Random<evo::Xoshiro256pp<evo::SplitMix64>> prng(s_seed);
	
	__m512i ranges = _mm512_set_epi64(
		0x510B3B098CE23BB6,
		0x0C30A0C0A580B8B4,
		0x0E4C2689DC2A5887,
		0x00000E4C2689DC2A,
		0x00000000D47ECD52,
		0x0000000059BDAF46,
		0x0000000036406D8C,
		0x000000000EEF1E41
	);

	__m512d sum = _mm512_setzero_pd();
	__m512i mx = _mm512_setzero_si512();
	__m512d target = _mm512_cvtepi64_pd(_mm512_srli_epi64(ranges, 1));
	target = _mm512_mul_pd(target, _mm512_set1_pd(s_iterations));

	SECTION("Without precomputed 't'")
	{
		for (int i = 0; i < s_iterations; i++)
		{
			__m512i crr = prng.range_512i(ranges);
			sum = _mm512_add_pd(sum, _mm512_cvtepi64_pd(crr));
			mx = _mm512_max_epi64(mx, crr);
		}
	}

	SECTION("With precomputed 't'")
	{
		__m512i t = prng.compute_t(ranges);
		for (int i = 0; i < s_iterations; i++)
		{
			__m512i crr = prng.range_512i(ranges, t);
			sum = _mm512_add_pd(sum, _mm512_cvtepi64_pd(crr));
			mx = _mm512_max_epi64(mx, crr);
		}
	}

	std::array<evo::f64_t, 8> sums{};
	_mm512_storeu_pd(sums.data(), sum);

	std::array<evo::f64_t, 8> targets{};
	_mm512_storeu_pd(targets.data(), target);

	for (int i = 0; i < 8; i++)
		REQUIRE_THAT(sums[i], Catch::Matchers::WithinRel(targets[i], 0.1));

	REQUIRE(_cvtmask8_u32(_mm512_cmplt_epu64_mask(mx, ranges)) == 0xFF);
}
