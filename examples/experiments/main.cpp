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
	return 0;
}
