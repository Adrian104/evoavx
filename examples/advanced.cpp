// This is an advanced example of how to use EvoAVX library.

#include <evoavx/evoavx.hpp>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// Fitness function is defined as a class.
class Rosenbrock
{
private:
	// Since the Rosenbrock function can take any number of arguments,
	// we store that number here.
	const evo::u64_t m_length;

public:
	Rosenbrock(evo::u64_t length)
		: m_length(length) {}

	// Returns the number of required arguments;
	// corresponds to the length of each chromosome.
	evo::u64_t length() const
	{
		return m_length;
	}

	// Main method that contains the optimized problem;
	// takes pointer to the chromosome (array of arguments)
	// and returns a score how well given arguments solve this problem.
	evo::f64_t evaluate(const evo::f64_t* args)
	{
		evo::f64_t sum = 0.0;
		for (evo::u64_t i = 0; i < m_length - 1; i++)
		{
			evo::f64_t p1 = args[i + 1] - args[i] * args[i];
			evo::f64_t p2 = 1.0 - args[i];

			sum += 100.0 * p1 * p1 + p2 * p2;
		}

		return sum;
	}
};

// Inspector is a class that allows to keep track of evolution progress.
// It must inherit from evo::Inspector and define at least inspect() method, which
// decides whether to continue or stop evolution on the current island.
// Moreover, it prints statistics and saves them to a file.
class AdvancedInspector : public evo::Inspector
{
private:
	const int m_rank;
	std::ofstream m_file;

public:
	AdvancedInspector(int rank)
		: m_rank(rank) {}

	// Runs once before the evolution begins.
	void init() override
	{
		m_file.open(std::string("island-") + std::to_string(m_rank) + ".csv");
		m_file << "generation,stagnation,max,min,mean,stdev,wtime\n";

		if (m_rank == 0)
		{
			std::cout << "\n Island with rank 0:\n\n";
			std::cout << "    gen  stag    maximum    minimum       mean      stdev      wtime\n";
			std::cout << " -------------------------------------------------------------------\n" << std::flush;
		}
	}

	evo::Action inspect(const evo::Statistics& stats) override
	{
		if (stats.m_generation == 1 || stats.m_generation % 10 == 0)
		{
			m_file << std::to_string(stats.m_generation) + ',' +
				std::to_string(stats.m_stagnation) + ',' +
				std::to_string(stats.m_maximum) + ',' +
				std::to_string(stats.m_minimum) + ',' +
				std::to_string(stats.m_mean) + ',' +
				std::to_string(stats.m_stdev) + ',' +
				std::to_string(stats.m_wtime) + '\n';
		}

		if (m_rank == 0 && (stats.m_generation == 1 || stats.m_generation % 100 == 0))
		{
			std::cout << std::setw(7) << std::right << stats.m_generation;
			std::cout << std::setw(6) << std::right << stats.m_stagnation;
			std::cout << std::setw(11) << std::scientific << std::setprecision(3) << stats.m_maximum;
			std::cout << std::setw(11) << std::scientific << std::setprecision(3) << stats.m_minimum;
			std::cout << std::setw(11) << std::scientific << std::setprecision(3) << stats.m_mean;
			std::cout << std::setw(11) << std::scientific << std::setprecision(3) << stats.m_stdev;
			std::cout << std::setw(11) << std::scientific << std::setprecision(3) << stats.m_wtime << std::endl;
		}

		// Evolution will run for at least 1 second (or more, if other islands are still busy).
		return stats.m_wtime < 1.0 ? evo::Action::CONTINUE : evo::Action::STOP;
	}

	// Runs once after the evolution finishes.
	void finish(const evo::Result& result) override
	{
		m_file.close();
		if (m_rank == 0)
		{
			std::cout << std::fixed << std::setprecision(8);
			std::cout << "\n Best solution found:\n";
			std::cout << " --------------------\n";
			std::cout << " Value: " << result.m_score;
			std::cout << "\n Arguments:\n  ";

			std::size_t c = 0;
			std::size_t last = result.m_args.size();

			for (evo::f64_t arg : result.m_args)
			{
				std::cout << ' ' << arg;

				if (++c < last)
					std::cout << ',';

				if (c % 5 == 0)
					std::cout << "\n  ";
			}

			std::cout << "\n\n Optimization took " << result.m_wtime << " seconds\n\n";
		}
	}
};

int main(int argc, char** argv)
{
	int size, rank;

	// MPI must be initialized manually.
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// In this example we use grid topology to connect islands.
	int rows = 1;
	int cols = size;

	for (int a = static_cast<int>(std::ceil(std::sqrt(size))); a > 0; a--)
	{
		if (size % a == 0)
		{
			rows = a;
			cols = size / a;
			break;
		}
	}

	std::vector<int> indices(size);
	std::vector<int> edges;

	for (int i = 0; i < size; i++)
	{
		int row = i / cols;
		int col = i % cols;
		std::vector<int> neighbors;

		if (row > 0)
			neighbors.push_back((row - 1) * cols + col);

		if (row < rows - 1)
			neighbors.push_back((row + 1) * cols + col);

		if (col > 0)
			neighbors.push_back(row * cols + (col - 1));

		if (col < cols - 1)
			neighbors.push_back(row * cols + (col + 1));

		edges.insert(edges.end(), neighbors.begin(), neighbors.end());
		indices[i] = edges.size();
	}

	// Static settings can be specified in a special evo::StaticSettings type.
	// First two parameters define the PRNG and its bounded integer generation algorithm.
	// evo::FusedXM option determines whether a combination of compatible crossover and mutation
	// will run as a single operator before writing a pack of genes to the RAM.
	// evo::ForceDomain option determines what should be done if some gene has a value outside its range.
	// evo::Cache option determines if a fitness function cache should be used.
#ifdef EVO_USE_IFMA52
	using settings = evo::StaticSettings<evo::Xoshiro256pp<evo::SplitMix64>, evo::RangeAlg::LEMIRE_52_FAST,
		evo::FusedXM::AUTO, evo::ForceDomain::AUTO, evo::Cache::ENABLED_CRC_32>;
#else
	using settings = evo::StaticSettings<evo::Xoshiro256pp<evo::SplitMix64>, evo::RangeAlg::LEMIRE_64_FAST,
		evo::FusedXM::AUTO, evo::ForceDomain::AUTO, evo::Cache::ENABLED_CRC_32>;
#endif

	// Main object of the evolution; represents a single island
	// (each MPI process must define its own evo::Evolution object)
	evo::Evolution<settings> evolution;

	// All islands need to have defined the same MPI_Comm with an associated graph.
	MPI_Comm graph;
	MPI_Graph_create(MPI_COMM_WORLD, size, indices.data(), edges.data(), 1, &graph);
	evolution.set_communicator(graph);

	// Since the Rosenbrock function can take any number of arguments,
	// we register 42 genes containing any value between -5.0 and 11.0.
	for (int i = 0; i < 42; i++)
		evolution.add_gene(-5.0, 11.0);

	// Registration of the inspector and the fitness function.
	// Both methods forward arguments to the constructors.
	evolution.set_inspector<AdvancedInspector>(rank);
	evolution.set_fitness_function<Rosenbrock>(evolution.get_genome().size());

	// evo::Extremum::MINIMUM means that the algorithm will solve a minimization problem
	// (lower score means better fitness)
	evolution.set_extremum(evo::Extremum::MINIMUM);

	// Seeds the internal PRNG. Note that the result may not be reproducible (despite the same seed)
	// if islands do not request migrations in the same order (e.g. due to varying computational speeds).
	evolution.set_seed(std::time(nullptr));

	// Sets the size of the cache to be 2^8 entries.
	evolution.set_cache_size_exponent(8);

	// EvoAVX supports heterogeneous islands. We can define different sets of operators
	// in a special evo::Blueprint type.
	using b0 = evo::Blueprint<evo::s::Tournament, evo::c::Linear, evo::m::Uniform>;
	using b1 = evo::Blueprint<evo::s::UnbiasedTournament, evo::c::BlendAlpha, evo::m::Boundary>;
	using b2 = evo::Blueprint<evo::s::SaRouletteSigma, evo::c::Uniform, evo::m::Uniform>;
	using b3 = evo::Blueprint<evo::s::SaRouletteWindow, evo::c::Average, evo::m::Uniform>;

	// Depending on the rank, algorithm picks corresponding operator set and configures them.
	switch (rank % 4)
	{
	case 0:
		evolution.set_blueprint<b0>();
		evolution.get_selection<b0>()->set_tournament_size(7);
		break;

	case 1:
		evolution.set_blueprint<b1>();
		evolution.get_crossover<b1>()->set_alpha(0.1);
		break;

	case 2:
		evolution.set_blueprint<b2>();
		evolution.get_selection<b2>()->set_parameters(0.1, 0.05, 2.0, 1e-4);
		break;

	case 3:
		evolution.set_blueprint<b3>();
		evolution.get_selection<b3>()->set_offset(0.5);
		break;
	}

	// Depending on the rank, algorithm picks corresponding hyperparameters.
	switch (rank % 3)
	{
	case 0:
		// set_population() takes 3 arguments:
		// first (400) - the total number of individuals in a single island;
		// second (400) - the number of individuals that pass selection operator and are able to reproduce;
		// third (50) - the number of migrants.
		evolution.set_population(400, 400, 50);

		// Crossover and mutation have a probability.
		evolution.set_crossover_probability(0.6);
		evolution.set_mutation_probability(0.1);

		// Maximum number of generations after which migration will be requested.
		evolution.set_migration_interval(30);
		break;

	case 1:
		evolution.set_population(340, 340, 10);
		evolution.set_crossover_probability(0.85);
		evolution.set_mutation_probability(0.02);
		evolution.set_migration_interval(20);
		break;

	case 2:
		evolution.set_population(375, 375, 20);
		evolution.set_crossover_probability(0.95);
		evolution.set_mutation_probability(0.005);
		evolution.set_migration_interval(15);
		break;
	}

	// Starts the evolution. Result is handled by the inspector's finish() method.
	evolution.run();

	MPI_Comm_free(&graph);
	MPI_Finalize();

	return 0;
}
