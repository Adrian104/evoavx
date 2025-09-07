#include <evoavx/evoavx.hpp>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

class Beale
{
public:
	evo::u64_t length() const
	{
		return 2;
	}

	evo::f64_t evaluate(const evo::f64_t* args)
	{
		evo::f64_t x = args[0];
		evo::f64_t y = args[1];

		evo::f64_t p1 = 1.5 - x + x * y;
		evo::f64_t p2 = 2.25 - x + x * y * y;
		evo::f64_t p3 = 2.625 - x + x * y * y * y;

		return p1 * p1 + p2 * p2 + p3 * p3;
	}
};

class BasicInspector : public evo::Inspector
{
public:
	evo::Action inspect(const evo::Statistics& stats) override
	{
		return stats.m_generation < 100 ? evo::Action::CONTINUE : evo::Action::STOP;
	}
};

int main(int argc, char** argv)
{
	int size, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<int> indices(size);
	std::vector<int> edges(size * 2);

	for (int i = 0; i < size; i++)
	{
		indices[i] = (i + 1) * 2;
		edges[i * 2] = (i - 1 + size) % size;
		edges[i * 2 + 1] = (i + 1) % size;
	}

	MPI_Comm graph;
	MPI_Graph_create(MPI_COMM_WORLD, size, indices.data(), edges.data(), 1, &graph);

	evo::Evolution evolution;

	evolution.add_gene(-5.0, 5.0);
	evolution.add_gene(-5.0, 5.0);
	evolution.set_inspector<BasicInspector>();
	evolution.set_fitness_function<Beale>();
	evolution.set_extremum(evo::Extremum::MINIMUM);
	evolution.set_blueprint<evo::Blueprint<evo::s::Tournament, evo::c::Arithmetic, evo::m::Uniform>>();
	evolution.set_crossover_probability(0.85);
	evolution.set_mutation_probability(0.1);
	evolution.set_population(50, 30, 5);
	evolution.set_migration_interval(12);
	evolution.set_seed(std::time(nullptr));
	evolution.set_communicator(graph);

	evo::Result result = evolution.run();

	if (rank == 0)
	{
		std::cout << std::fixed << std::setprecision(8);
		std::cout << "\n Best solution found:\n";
		std::cout << " --------------------\n";
		std::cout << "       x = " << result.m_args[0] << '\n';
		std::cout << "       y = " << result.m_args[1] << '\n';
		std::cout << " f(x, y) = " << result.m_score << '\n';
		std::cout << "\n Optimization took " << result.m_wtime << " seconds\n\n";
	}

	MPI_Comm_free(&graph);
	MPI_Finalize();

	return 0;
}
