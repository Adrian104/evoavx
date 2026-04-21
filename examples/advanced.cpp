#include <evoavx/evoavx.hpp>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

class Rosenbrock
{
private:
	const evo::u64_t m_length;

public:
	Rosenbrock(evo::u64_t length)
		: m_length(length) {}

	evo::u64_t length() const
	{
		return m_length;
	}

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

class AdvancedInspector : public evo::Inspector
{
private:
	const int m_rank;
	std::ofstream m_file;

public:
	AdvancedInspector(int rank)
		: m_rank(rank) {}

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

		return stats.m_wtime < 1.0 ? evo::Action::CONTINUE : evo::Action::STOP;
	}

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

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

	MPI_Comm graph;
	MPI_Graph_create(MPI_COMM_WORLD, size, indices.data(), edges.data(), 1, &graph);

#ifdef EVO_USE_IFMA52
	using settings = evo::StaticSettings<evo::Xoshiro256pp<evo::SplitMix64>, evo::RangeAlg::LEMIRE_52_FAST,
		evo::FusedXM::AUTO, evo::ForceDomain::AUTO, evo::Cache::ENABLED_CRC_32>;
#else
	using settings = evo::StaticSettings<evo::Xoshiro256pp<evo::SplitMix64>, evo::RangeAlg::LEMIRE_64_FAST,
		evo::FusedXM::AUTO, evo::ForceDomain::AUTO, evo::Cache::ENABLED_CRC_32>;
#endif

	evo::Evolution<settings> evolution;
	for (int i = 0; i < 42; i++)
		evolution.add_gene(-5.0, 11.0);

	evolution.set_inspector<AdvancedInspector>(rank);
	evolution.set_fitness_function<Rosenbrock>(evolution.get_genome().size());
	evolution.set_extremum(evo::Extremum::MINIMUM);
	evolution.set_seed(std::time(nullptr));
	evolution.set_communicator(graph);
	evolution.set_cache_size_exponent(8);

	using b0 = evo::Blueprint<evo::s::Tournament, evo::c::Linear, evo::m::Uniform>;
	using b1 = evo::Blueprint<evo::s::UnbiasedTournament, evo::c::BlendAlpha, evo::m::Boundary>;
	using b2 = evo::Blueprint<evo::s::SaRouletteSigma, evo::c::Uniform, evo::m::Uniform>;
	using b3 = evo::Blueprint<evo::s::SaRouletteWindow, evo::c::Average, evo::m::Uniform>;

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

	switch (rank % 3)
	{
	case 0:
		evolution.set_population(400, 400, 50);
		evolution.set_crossover_probability(0.6);
		evolution.set_mutation_probability(0.1);
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

	evolution.run();

	MPI_Comm_free(&graph);
	MPI_Finalize();

	return 0;
}
