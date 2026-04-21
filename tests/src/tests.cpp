#include "pch.hpp"
#include "test-component.hpp"
#include "test-global.hpp"
#include "test-random.hpp"
#include "test-shuffle.hpp"
#include "test-statistics.hpp"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int result = Catch::Session().run(argc, argv);

	MPI_Finalize();
	return result;
}
