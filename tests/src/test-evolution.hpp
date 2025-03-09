#include "pch.hpp"
#include <evoavx/evoavx.hpp>

class MyFitnessFunction
{
	evo::u64_t m_length;

public:
	MyFitnessFunction(evo::u64_t length)
		: m_length(length) {}

	evo::u64_t length() const { return m_length; }
	evo::f64_t evaluate([[maybe_unused]] const evo::f64_t* genes) { return 1.0; }
};

class MyFitnessFunction2
{
	evo::u64_t m_length;

public:
	MyFitnessFunction2(evo::u64_t length)
		: m_length(length) {}

	evo::u64_t length() const { return m_length; }
	evo::f64_t evaluate([[maybe_unused]] const evo::f64_t* genes) { return 2.0; }
};

template <evo::cc::static_settings S> struct MySelection { int x; };
template <evo::cc::static_settings S> struct MyCrossover { bool y; };
template <evo::cc::static_settings S> struct MyMutation { char z; };
template <evo::cc::static_settings S> struct MyMutation2 { float w; };

TEST_CASE("Evolution getters and setters work as expected")
{
	constexpr evo::f64_t margin = 1e-8;
	constexpr evo::u64_t islands = 4;
	constexpr evo::u64_t u64max = std::numeric_limits<evo::u64_t>::max();

	evo::Evolution ev(islands);

	REQUIRE(ev.get_genome().empty());
	REQUIRE(ev.get_island_count() == islands);

	for (evo::u64_t i = 0; i < islands; i++)
	{
		REQUIRE(ev.island_get_population(i).first == 0);
		REQUIRE(ev.island_get_population(i).second == 0);
	}

	REQUIRE_THROWS(ev.island_get_population(islands));
	REQUIRE_THROWS(ev.island_get_population(u64max));

	ev.add_gene(3.0, 7.0);
	ev.clear_genome();
	ev.add_gene(5.0, 32.0);
	ev.add_gene(61.0, 2.0);
	ev.add_gene(-5.0, -9.0);
	ev.set_seed(42);
	ev.set_extremum(evo::Extremum::MAXIMUM);
	ev.set_population(500, 300);
	ev.set_population(333, 104);
	ev.set_crossover_probability(0.75);
	ev.set_mutation_probability(0.08);

	REQUIRE_THROWS(ev.set_population(0, 0));
	REQUIRE_THROWS(ev.set_population(55, 211));
	REQUIRE_THROWS(ev.set_crossover_probability(-0.5));
	REQUIRE_THROWS(ev.set_crossover_probability(1.5));
	REQUIRE_THROWS(ev.set_mutation_probability(-0.5));
	REQUIRE_THROWS(ev.set_mutation_probability(1.5));

	ev.island_set_extremum(0, evo::Extremum::MINIMUM);
	ev.island_set_population(1, 987, 654);
	ev.island_set_crossover_probability(2, 0.4);
	ev.island_set_mutation_probability(3, 0.6);

	REQUIRE_THROWS(ev.island_set_population(1, 0, 0));
	REQUIRE_THROWS(ev.island_set_population(1, 55, 211));
	REQUIRE_THROWS(ev.island_set_crossover_probability(2, -0.5));
	REQUIRE_THROWS(ev.island_set_crossover_probability(2, 1.5));
	REQUIRE_THROWS(ev.island_set_mutation_probability(3, -0.5));
	REQUIRE_THROWS(ev.island_set_mutation_probability(3, 1.5));

	REQUIRE_THROWS(ev.island_set_extremum(islands, evo::Extremum::MAXIMUM));
	REQUIRE_THROWS(ev.island_set_population(islands, 999, 888));
	REQUIRE_THROWS(ev.island_set_crossover_probability(islands, 0.9));
	REQUIRE_THROWS(ev.island_set_mutation_probability(islands, 0.9));

	using MyTriplet = evo::Triplet<MySelection, MyCrossover, MyMutation>;
	using MyTriplet2 = evo::Triplet<MySelection, MyCrossover, MyMutation2>;

	ev.set_fitness_function<MyFitnessFunction>(73);
	ev.set_triplet<MyTriplet>();
	ev.remove_fitness_function<MyFitnessFunction>();
	ev.remove_triplet<MyTriplet>();
	ev.set_fitness_function<MyFitnessFunction>(55);
	ev.set_triplet<MyTriplet2>();

	ev.island_set_fitness_function<MyFitnessFunction>(0, 1111);
	ev.island_set_fitness_function<MyFitnessFunction2>(1, 2222);
	ev.island_set_triplet<MyTriplet>(2);
	ev.island_remove_fitness_function<MyFitnessFunction>(3);
	ev.island_remove_triplet<MyTriplet2>(3);

	std::vector<std::pair<evo::f64_t, evo::f64_t>> expected;
	std::vector<std::pair<evo::f64_t, evo::f64_t>> real = ev.get_genome();

	expected.emplace_back(5.0, 32.0);
	expected.emplace_back(2.0, 61.0);
	expected.emplace_back(-9.0, -5.0);

	REQUIRE(expected.size() == real.size());
	for (evo::u64_t i = 0; i < 3; i++)
	{
		auto& [ea, eb] = expected[i];
		auto& [ra, rb] = real[i];

		REQUIRE_THAT(ra, Catch::Matchers::WithinAbs(ea, margin));
		REQUIRE_THAT(rb, Catch::Matchers::WithinAbs(eb, margin));
	}

	REQUIRE(ev.island_get_population(2).first == 333);
	REQUIRE(ev.island_get_population(0).second == 104);
	REQUIRE(ev.island_get_population(1).first == 987);
	REQUIRE(ev.island_get_population(1).second == 654);

	REQUIRE_THAT(ev.island_get_crossover_probability(0), Catch::Matchers::WithinAbs(0.75, margin));
	REQUIRE_THAT(ev.island_get_crossover_probability(2), Catch::Matchers::WithinAbs(0.4, margin));
	REQUIRE_THAT(ev.island_get_mutation_probability(1), Catch::Matchers::WithinAbs(0.08, margin));
	REQUIRE_THAT(ev.island_get_mutation_probability(3), Catch::Matchers::WithinAbs(0.6, margin));

	REQUIRE(ev.island_get_extremum(0) == evo::Extremum::MINIMUM);
	REQUIRE(ev.island_get_extremum(1) == evo::Extremum::MAXIMUM);
	REQUIRE(ev.get_island_count() == islands);
	REQUIRE(ev.get_seed() == 42);

	REQUIRE_THROWS(ev.island_get_population(islands));
	REQUIRE_THROWS(ev.island_get_crossover_probability(islands));
	REQUIRE_THROWS(ev.island_get_mutation_probability(islands));
	REQUIRE_THROWS(ev.island_get_extremum(islands));

	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction>(0) != nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction>(1) != nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction>(2) != nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction>(3) == nullptr);

	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(0) == nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(1) != nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(2) == nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(3) == nullptr);

	ev.island_get_selection<MyTriplet2>(0)->x = -9;
	ev.island_get_crossover<MyTriplet2>(1)->y = true;
	ev.island_get_mutation<MyTriplet>(2)->z = '*';

	REQUIRE(ev.island_get_selection<MyTriplet2>(0) != nullptr);
	REQUIRE(ev.island_get_crossover<MyTriplet2>(1) != nullptr);
	REQUIRE(ev.island_get_mutation<MyTriplet2>(2) != nullptr);
	REQUIRE(ev.island_get_selection<MyTriplet2>(3) == nullptr);

	REQUIRE(ev.island_get_crossover<MyTriplet>(0) == nullptr);
	REQUIRE(ev.island_get_mutation<MyTriplet>(1) == nullptr);
	REQUIRE(ev.island_get_selection<MyTriplet>(2) != nullptr);
	REQUIRE(ev.island_get_crossover<MyTriplet>(3) == nullptr);

	REQUIRE(ev.island_get_selection<MyTriplet2>(0)->x == -9);
	REQUIRE(ev.island_get_crossover<MyTriplet2>(1)->y == true);
	REQUIRE(ev.island_get_mutation<MyTriplet>(2)->z == '*');

	REQUIRE_THROWS(ev.island_get_fitness_function<MyFitnessFunction>(islands));
	REQUIRE_THROWS(ev.island_get_selection<MyTriplet2>(islands));
	REQUIRE_THROWS(ev.island_get_crossover<MyTriplet2>(islands));
	REQUIRE_THROWS(ev.island_get_mutation<MyTriplet2>(islands));
}
