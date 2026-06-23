# EvoAVX
EvoAVX is an open-source C++ library implementing an island-based genetic algorithm, designed for CPUs with AVX-512 instructions. It focuses on speed by minimizing GA overhead. Parallelism is achieved via MPI, with each rank corresponding to a single island.

## Building
**Prerequisites:**

- C++20 compiler
- CMake 3.25 (or newer)
- MPI 4.0 implementation (e.g., MPICH)

To generate project files, simply run the following in the top level directory:
```bash
cmake -S . -B build
```

You can optionally specify the following settings:

- `EVOAVX_BUILD_EXAMPLES` - enables building example programs.
- `EVOAVX_BUILD_TESTS` - enables building unit tests; Catch2 framework will be automatically downloaded if not present on the system.
- `EVOAVX_ENABLE_IFMA52` - enables optimizations that utilize `IFMA52` instructions from the `AVX-512` family.

If the repository is not nested inside another project, both examples and tests will be built by default.

After building the project with specified build system (e.g., make) you can now run examples as follows:
```bash
mpiexec -n X ./build/examples/evoavx-basic
mpiexec -n X ./build/examples/evoavx-advanced
```
where `X` is the number of islands. If unsure, set `X` to the number of cores your CPU has available.

To run unit tests:
```bash
./build/tests/evoavx-tests
```

## Usage
In order to perform an optimization task, the user needs to:

1. initialize MPI,
2. create a custom MPI communicator with an associated graph (island topology is defined manually),
3. create `evo::Evolution` instance,
4. configure hyperparameters, specify a fitness function and other options (including binding previously created communicator),
5. call `run()` method,
6. clean up the communicator and finalize the MPI environment.

The best solution found across all islands will be returned by the `run()` method. If an inspector was set beforehand, it will receive statistics during optimization for each island individually.

For more information on how to use EvoAVX, see `examples` directory.

## How to cite EvoAVX
Please use the following BibTeX entry to cite EvoAVX in scientific publications:
```
@ARTICLE{11343780,
  author={Kulawik, Adrian and Krużel, Filip},
  journal={IEEE Access},
  title={EvoAVX: Island-Based Genetic Algorithm Library With AVX-512 and MPI},
  year={2026},
  volume={14},
  number={},
  pages={6944-6953},
  keywords={Genetic algorithms;Libraries;Single instruction multiple data;C++ languages;Topology;Registers;Genetic operators;Performance evaluation;Instruction sets;Generators;AVX-512;evolutionary computation;genetic algorithms;high-performance computing;island model;MPI;parallel computing;SIMD vectorization;Xoshiro256++},
  doi={10.1109/ACCESS.2026.3652358}}
```
