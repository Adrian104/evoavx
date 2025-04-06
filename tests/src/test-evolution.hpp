#include "pch.hpp"
#include <evoavx/evoavx.hpp>

class MyFitnessFunction
{
	evo::u64_t m_length;

public:
	MyFitnessFunction(evo::u64_t length = 55)
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

template <evo::cc::static_settings S>
struct MyCrossover
{
	constexpr static bool s_twins = true;
	constexpr static bool s_fusedXM = true;
	constexpr static bool s_forceDomain = true;

	bool y;

	void init_generation(evo::Island<S>& island) {}
	void init_wave(evo::Island<S>& island) {}

	template <typename MutationT>
	void perform(evo::Island<S>& island, evo::f64_t* a, evo::f64_t* b, evo::f64_t* c) {}
};

template <evo::cc::static_settings S>
struct MyMutation
{
	constexpr static bool s_fusedXM = true;

	char z;

	void init_generation(evo::Island<S>& island) {}
	void init_wave(evo::Island<S>& island) {}

	template <bool forceDomain>
	void perform(evo::Island<S>& island, evo::f64_t* a) {}

	template <bool forceDomain>
	static __m512d perform(__m512d genes, __m512d min, __m512d max, evo::f64_t prob, evo::Random<S>& rand) { return _mm512_setzero_pd(); }
};

template <evo::cc::static_settings S>
struct MyMutation2
{
	constexpr static bool s_fusedXM = false;

	float w;

	void init_generation(evo::Island<S>& island) {}
	void init_wave(evo::Island<S>& island) {}

	template <bool forceDomain>
	void perform(evo::Island<S>& island, evo::f64_t* a) {}

	template <bool forceDomain>
	static __m512d perform(__m512d genes, __m512d min, __m512d max, evo::f64_t prob, evo::Random<S>& rand) { return _mm512_setzero_pd(); }
};

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

	using MyBlueprint = evo::Blueprint<evo::s::Tournament, MyCrossover, MyMutation>;
	using MyBlueprint2 = evo::Blueprint<evo::s::Tournament, MyCrossover, MyMutation2>;

	ev.set_fitness_function<MyFitnessFunction>();
	ev.set_blueprint<MyBlueprint>();

	ev.island_set_fitness_function<MyFitnessFunction>(0, 1111);
	ev.island_set_fitness_function<MyFitnessFunction2>(1, 2222);
	ev.island_set_blueprint<MyBlueprint>(2);
	ev.island_set_blueprint<MyBlueprint2>(3);

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

	for (evo::u64_t i = 0; i < 4; i++)
	{
		REQUIRE(ev.island_get_fitness_function<MyFitnessFunction>(i) != nullptr);
		REQUIRE(ev.island_get_selection<MyBlueprint>(i) != nullptr);
		REQUIRE(ev.island_get_crossover<MyBlueprint>(i) != nullptr);
		REQUIRE(ev.island_get_mutation<MyBlueprint>(i) != nullptr);
	}

	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(0) == nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(1) != nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(2) == nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(3) == nullptr);
	REQUIRE(ev.island_get_fitness_function<MyFitnessFunction2>(1)->length() == 2222);

	REQUIRE(ev.island_is_fitness_function_set<MyFitnessFunction>(0));
	REQUIRE_FALSE(ev.island_is_fitness_function_set<MyFitnessFunction>(1));
	REQUIRE(ev.island_is_fitness_function_set<MyFitnessFunction>(2));
	REQUIRE(ev.island_is_fitness_function_set<MyFitnessFunction>(3));

	REQUIRE_FALSE(ev.island_is_fitness_function_set<MyFitnessFunction2>(0));
	REQUIRE(ev.island_is_fitness_function_set<MyFitnessFunction2>(1));
	REQUIRE_FALSE(ev.island_is_fitness_function_set<MyFitnessFunction2>(2));
	REQUIRE_FALSE(ev.island_is_fitness_function_set<MyFitnessFunction2>(3));

	REQUIRE(ev.island_get_selection<MyBlueprint2>(0) == nullptr);
	REQUIRE(ev.island_get_crossover<MyBlueprint2>(1) == nullptr);
	REQUIRE(ev.island_get_mutation<MyBlueprint2>(2) == nullptr);
	REQUIRE(ev.island_get_selection<MyBlueprint2>(3) != nullptr);

	REQUIRE(ev.island_is_blueprint_set<MyBlueprint>(0));
	REQUIRE(ev.island_is_blueprint_set<MyBlueprint>(1));
	REQUIRE(ev.island_is_blueprint_set<MyBlueprint>(2));
	REQUIRE_FALSE(ev.island_is_blueprint_set<MyBlueprint>(3));

	REQUIRE_FALSE(ev.island_is_blueprint_set<MyBlueprint2>(0));
	REQUIRE_FALSE(ev.island_is_blueprint_set<MyBlueprint2>(1));
	REQUIRE_FALSE(ev.island_is_blueprint_set<MyBlueprint2>(2));
	REQUIRE(ev.island_is_blueprint_set<MyBlueprint2>(3));

	REQUIRE_THROWS(ev.island_get_fitness_function<MyFitnessFunction>(islands));
	REQUIRE_THROWS(ev.island_get_selection<MyBlueprint2>(islands));
	REQUIRE_THROWS(ev.island_get_crossover<MyBlueprint2>(islands));
	REQUIRE_THROWS(ev.island_get_mutation<MyBlueprint2>(islands));
	REQUIRE_THROWS(ev.island_is_fitness_function_set<MyFitnessFunction>(islands));
	REQUIRE_THROWS(ev.island_is_blueprint_set<MyBlueprint2>(islands));
}
