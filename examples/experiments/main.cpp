#include <evoavx/evoavx.hpp>
#include <cassert>

class Sphere : public evo::FitnessFunction
{
	evo::u64_t m_dim;

public:
	Sphere(evo::u64_t dim) : m_dim(dim) {}

	evo::u64_t length() const override;
	evo::f64_t evaluate(const evo::f64_t* genes, evo::u64_t threadId) override;
};

template <evo::cc::static_settings S>
class SimpleSelection : public evo::Selection<S>
{

};

template <evo::cc::static_settings S>
class SimpleCrossover
{

};

template <evo::cc::static_settings S>
class SimpleMutation
{

};

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

	assert(evolution.get<Sphere>() == nullptr);
	assert(evolution.get<SimpleSelection>() == nullptr);
	assert((evolution.get<SimpleCrossover, SimpleMutation>() == nullptr));

	evolution.set<Sphere>(10);
	evolution.set<SimpleSelection>();
	evolution.set<SimpleCrossover, SimpleMutation>();

	assert(evolution.get<Sphere>() != nullptr);
	assert(evolution.get<Sphere>()->length() == 10);
	assert(evolution.get<SimpleSelection>() != nullptr);
	assert((evolution.get<SimpleCrossover, SimpleMutation>() != nullptr));

	evolution.set_crossover_probability(0.5);
	evolution.set_population(500, 300);

	return 0;
}
