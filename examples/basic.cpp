// This is a basic example of how to use EvoAVX library.

#include <evoavx/evoavx.hpp>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

// Fitness function is defined as a class.
class Beale
{
public:
	// Returns the number of required arguments;
	// corresponds to the length of each chromosome.
	evo::u64_t length() const
	{
		return 2;
	}

	// Main method that contains the optimized problem;
	// takes pointer to the chromosome (array of arguments)
	// and returns a score how well given arguments solve this problem.
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

// Inspector is a class that allows to keep track of evolution progress.
// It must inherit from evo::Inspector and define at least inspect() method, which
// decides whether to continue or stop evolution on the current island.
class BasicInspector : public evo::Inspector
{
public:
	evo::Action inspect(const evo::Statistics& stats) override
	{
		// Runs for at least 100 generations (or more, if other islands are still busy).
		return stats.m_generation < 100 ? evo::Action::CONTINUE : evo::Action::STOP;
	}
};

int main(int argc, char** argv)
{
	int size, rank;

	// MPI must be initialized manually.
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// In this example we use ring topology to connect islands.
	std::vector<int> indices(size);
	std::vector<int> edges(size * 2);

	for (int i = 0; i < size; i++)
	{
		indices[i] = (i + 1) * 2;
		edges[i * 2] = (i - 1 + size) % size;
		edges[i * 2 + 1] = (i + 1) % size;
	}

	// Main object of the evolution; represents a single island
	// (each MPI process must define its own evo::Evolution object)
	evo::Evolution evolution;

	// All islands need to have defined the same MPI_Comm with an associated graph.
	MPI_Comm graph;
	MPI_Graph_create(MPI_COMM_WORLD, size, indices.data(), edges.data(), 1, &graph);
	evolution.set_communicator(graph);

	// Since Beale function takes 2 arguments, we register 2 genes
	// each containing any value between -5.0 and 5.0.
	evolution.add_gene(-5.0, 5.0);
	evolution.add_gene(-5.0, 5.0);

	// Registration of the inspector and the fitness function.
	evolution.set_inspector<BasicInspector>();
	evolution.set_fitness_function<Beale>();

	// evo::Extremum::MINIMUM means that the algorithm will solve a minimization problem
	// (lower score means better fitness)
	evolution.set_extremum(evo::Extremum::MINIMUM);

	// Selection (in namespace evo::s::), crossover (evo::c::), and mutation (evo::m::)
	// must all be defined in a special evo::Blueprint type.
	evolution.set_blueprint<evo::Blueprint<evo::s::Tournament, evo::c::Arithmetic, evo::m::Uniform>>();

	// Crossover and mutation have a probability.
	evolution.set_crossover_probability(0.85);
	evolution.set_mutation_probability(0.1);

	// set_population() takes 3 arguments:
	// first (50) - the total number of individuals in a single island;
	// second (30) - the number of individuals that pass selection operator and are able to reproduce;
	// third (5) - the number of migrants.
	evolution.set_population(50, 30, 5);

	// Maximum number of generations after which migration will be requested.
	evolution.set_migration_interval(12);

	// Seeds the internal PRNG. Note that the result may not be reproducible (despite the same seed)
	// if islands do not request migrations in the same order (e.g. due to varying computational speeds).
	evolution.set_seed(std::time(nullptr));

	// Starts the evolution; returns global result (reduced across all islands).
	evo::Result result = evolution.run();

	// Only a single rank prints the global result.
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
