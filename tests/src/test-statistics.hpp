#include "pch.hpp"
#include <evoavx/evoavx.hpp>

/*
TEST_CASE("Statistics are calculated with good accuracy")
{
	evo::State<evo::DefaultStaticSettings> state;
	evo::Statistics<evo::DefaultStaticSettings> stats;

	SECTION("Tiny data set")
	{
		constexpr evo::f64_t margin = 1e-8;

		state.m_indivCount = 3;
		state.m_scores = evo::unique<evo::f64_t[]>(evo::allocate<evo::f64_t>(8));

		state.m_scores[0] = std::numbers::pi;
		state.m_scores[1] = std::numbers::phi;
		state.m_scores[2] = -std::numbers::e;

		stats.start();
		stats.update(state, 0.68);

		REQUIRE_THAT(stats.m_mean, Catch::Matchers::WithinAbs(0.6804482713, margin));
		REQUIRE_THAT(stats.m_stdev, Catch::Matchers::WithinAbs(2.4824493912, margin));
		REQUIRE_THAT(stats.m_minimum, Catch::Matchers::WithinAbs(-std::numbers::e, margin));
		REQUIRE_THAT(stats.m_maximum, Catch::Matchers::WithinAbs(std::numbers::pi, margin));

		REQUIRE(stats.m_minimumPos == 2);
		REQUIRE(stats.m_maximumPos == 0);
		REQUIRE(stats.m_generation == 1);
		REQUIRE(stats.m_stagnation == 0);
	}

	SECTION("Tiny data set (large deviation from mean estimate)")
	{
		constexpr evo::f64_t margin = 1e-6;

		state.m_indivCount = 3;
		state.m_scores = evo::unique<evo::f64_t[]>(evo::allocate<evo::f64_t>(8));

		state.m_scores[0] = std::numbers::pi;
		state.m_scores[1] = std::numbers::phi;
		state.m_scores[2] = -std::numbers::e;

		stats.start();
		stats.update(state, 1234.56789);

		REQUIRE_THAT(stats.m_mean, Catch::Matchers::WithinAbs(0.6804482713, margin));
		REQUIRE_THAT(stats.m_stdev, Catch::Matchers::WithinAbs(2.4824493912, margin));
		REQUIRE_THAT(stats.m_minimum, Catch::Matchers::WithinAbs(-std::numbers::e, margin));
		REQUIRE_THAT(stats.m_maximum, Catch::Matchers::WithinAbs(std::numbers::pi, margin));

		REQUIRE(stats.m_minimumPos == 2);
		REQUIRE(stats.m_maximumPos == 0);
		REQUIRE(stats.m_generation == 1);
		REQUIRE(stats.m_stagnation == 0);
	}

	SECTION("Small data set")
	{
		constexpr evo::f64_t margin = 1e-8;

		state.m_indivCount = 25;
		state.m_scores = evo::unique<evo::f64_t[]>(evo::allocate<evo::f64_t>(32));

		for (evo::i32_t i = 0; i < 5; i++)
		{
			evo::f64_t x = static_cast<evo::f64_t>(i);

			state.m_scores[(evo::u64_t)i * 5 + 0] = std::numbers::phi * x;
			state.m_scores[(evo::u64_t)i * 5 + 1] = std::numbers::pi * (x + 1.0);
			state.m_scores[(evo::u64_t)i * 5 + 2] = -std::numbers::sqrt2 * x;
			state.m_scores[(evo::u64_t)i * 5 + 3] = -std::numbers::inv_sqrtpi;
			state.m_scores[(evo::u64_t)i * 5 + 4] = std::numbers::e * x * x;
		}

		stats.start();
		stats.update(state, 5.12);

		REQUIRE_THAT(stats.m_mean, Catch::Matchers::WithinAbs(5.1155840401, margin));
		REQUIRE_THAT(stats.m_stdev, Catch::Matchers::WithinAbs(10.2848561845, margin));
		REQUIRE_THAT(stats.m_minimum, Catch::Matchers::WithinAbs(-5.6568542495, margin));
		REQUIRE_THAT(stats.m_maximum, Catch::Matchers::WithinAbs(43.4925092553, margin));

		REQUIRE(stats.m_minimumPos == 22);
		REQUIRE(stats.m_maximumPos == 24);
		REQUIRE(stats.m_generation == 1);
		REQUIRE(stats.m_stagnation == 0);
	}

	SECTION("Large data set")
	{
		state.m_indivCount = 1ULL << 20;
		state.m_scores = evo::unique<evo::f64_t[]>(evo::allocate<evo::f64_t>(evo::alignment_ceil<evo::f64_t>(state.m_indivCount)));

		std::mt19937_64 mt(42);
		std::uniform_real_distribution<double> uniform(-std::numbers::pi * 1e5, std::numbers::e * 1e6);

		for (evo::u64_t i = 0; i < state.m_indivCount; i++)
			state.m_scores[i] = uniform(mt);

		stats.start();
		stats.update(state, 1202000.0);

		REQUIRE_THAT(stats.m_mean, Catch::Matchers::WithinRel(1202061.282, 0.01));
		REQUIRE_THAT(stats.m_stdev, Catch::Matchers::WithinRel(875390.3409, 0.01));
		REQUIRE_THAT(stats.m_minimum, Catch::Matchers::WithinRel(-std::numbers::pi * 1e5, 0.01));
		REQUIRE_THAT(stats.m_maximum, Catch::Matchers::WithinRel(std::numbers::e * 1e6, 0.01));
	}
}
*/