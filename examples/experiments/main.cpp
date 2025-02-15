#include <evoavx/evoavx.hpp>
#include <cassert>
#include <cmath>

class Sphere : public evo::FitnessFunction
{
	evo::u64_t m_dim;

public:
	Sphere(evo::u64_t dim) : m_dim(dim) {}

	evo::u64_t length() const override;
	evo::f64_t evaluate(const evo::f64_t* genes, evo::u64_t threadId) override;
};

template <evo::cc::static_settings S>
class Sel
{
public:
	int x = 3;
};

template <evo::cc::static_settings S>
class Cross
{
public:
	int y = 4;
};

template <evo::cc::static_settings S>
class Mut
{
public:
	int z = 5;
};

template <evo::cc::static_settings S>
class Sel2
{
public:
	int a = 30;
};

template <evo::cc::static_settings S>
class Cross2
{
public:
	int b = 40;
};

template <evo::cc::static_settings S>
class Mut2
{
public:
	int c = 50;
};

using TripletA = evo::Triplet<Sel, Cross, Mut>;
using TripletB = evo::Triplet<Sel, Cross2, Mut>;
using TripletC = evo::Triplet<Sel2, Cross, Mut2>;

evo::u64_t Sphere::length() const
{
	return m_dim;
}

evo::f64_t Sphere::evaluate(const evo::f64_t* genes, evo::u64_t threadId)
{
	evo::f64_t sum = 0;
	for (evo::u64_t i = 0; i < m_dim; i++)
		sum += genes[i] * genes[i];

	return sum;
}

int main()
{
	evo::Evolution evolution;

	assert(evolution.get_fitness_function<Sphere>() == nullptr);
	assert(evolution.get_selection<TripletA>() == nullptr);
	assert(evolution.get_crossover<TripletB>() == nullptr);
	assert(evolution.get_mutation<TripletC>() == nullptr);

	for (int i = 1; i <= 10; i++)
		evolution.add_gene(0, std::pow(10, i / 2.0));

	evolution.set_extremum(evo::Extremum::MAXIMUM);
	evolution.set_population(799, 401);
	evolution.set_crossover_probability(0.8);
	evolution.set_mutation_probability(0.15);

	assert(evolution.get_genome().size() == 10);
	assert(evolution.get_population().first == 799);
	assert(evolution.get_population().second == 401);

	evolution.set_fitness_function<Sphere>(10);
	evolution.set_triplet<TripletA>();
	evolution.set_triplet<TripletB>();

	assert(evolution.get_selection<TripletA>() != nullptr);
	assert(evolution.get_crossover<TripletA>() != nullptr);
	assert(evolution.get_mutation<TripletA>() != nullptr);

	assert(evolution.get_selection<TripletB>() != nullptr);
	assert(evolution.get_crossover<TripletB>() != nullptr);
	assert(evolution.get_mutation<TripletB>() != nullptr);

	assert(evolution.get_selection<TripletB>()->x == 3);
	assert(evolution.get_crossover<TripletB>()->b == 40);
	assert(evolution.get_mutation<TripletB>()->z == 5);

	assert(evolution.get_selection<TripletC>() == nullptr);
	assert(evolution.get_crossover<TripletC>() == nullptr);
	assert(evolution.get_mutation<TripletC>() == nullptr);

	evolution.run();
	return 0;
}
