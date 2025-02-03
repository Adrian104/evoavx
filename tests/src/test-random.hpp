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

TEST_CASE("Method range_512i_52(u64_t) works as expected")
{
	constexpr static evo::u64_t s_seed = 42;
	constexpr static int s_iterations = 1 << 10;

	evo::Random<evo::Xoshiro256pp<evo::SplitMix64>> prng(s_seed);
	evo::u64_t range = 1ULL << 50;

	__m512i sum = _mm512_setzero_si512();
	__m512i max = _mm512_set1_epi64(0);

	for (int i = 0; i < s_iterations; i++)
	{
		__m512i crr = prng.range_512i_52(range);
		sum = _mm512_add_epi64(sum, crr);
		max = _mm512_max_epi64(max, crr);
	}

	std::array<evo::u64_t, 8> sumArr{};
	std::array<evo::u64_t, 8> maxArr{};

	_mm512_storeu_epi64(sumArr.data(), sum);
	_mm512_storeu_epi64(maxArr.data(), max);

	double target = static_cast<double>(range >> 1);

	for (auto v : sumArr)
		REQUIRE_THAT(static_cast<double>(v) / s_iterations, Catch::Matchers::WithinRel(target, 0.1));

	for (auto v : maxArr)
		REQUIRE(v < range);
}

TEST_CASE("Method range_512i_52(__m512i) works as expected")
{
	constexpr static evo::u64_t s_seed = 42;
	constexpr static int s_iterations = 1 << 10;

	evo::Random<evo::Xoshiro256pp<evo::SplitMix64>> prng(s_seed);

	__m512i range = _mm512_set_epi64(1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47, 1ULL << 48, 1ULL << 49, 1ULL << 50);
	__m512i sum = _mm512_setzero_si512();
	__m512i max = _mm512_set1_epi64(0);

	for (int i = 0; i < s_iterations; i++)
	{
		__m512i crr = prng.range_512i_52(range);
		sum = _mm512_add_epi64(sum, crr);
		max = _mm512_max_epi64(max, crr);
	}

	std::array<evo::u64_t, 8> sumArr{};
	std::array<evo::u64_t, 8> maxArr{};

	_mm512_storeu_epi64(sumArr.data(), sum);
	_mm512_storeu_epi64(maxArr.data(), max);

	for (evo::u64_t i = 0; i < 8; i++)
	{
		evo::u64_t mx = 1ULL << (50 - i);
		double target = static_cast<double>(1ULL << (49 - i));

		REQUIRE_THAT(static_cast<double>(sumArr[i]) / s_iterations, Catch::Matchers::WithinRel(target, 0.1));
		REQUIRE(maxArr[i] < mx);
	}
}
