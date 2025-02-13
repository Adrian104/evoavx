cmake_minimum_required(VERSION 3.25)
project(evoavx-root LANGUAGES NONE)

option(EVOAVX_BUILD_TESTS "Build unit tests for EvoAVX" ${PROJECT_IS_TOP_LEVEL})
option(EVOAVX_BUILD_EXAMPLES "Build examples for EvoAVX" ${PROJECT_IS_TOP_LEVEL})

add_subdirectory(core)

if(${EVOAVX_BUILD_TESTS})
	add_subdirectory(tests)
endif()

if(${EVOAVX_BUILD_EXAMPLES})
	add_subdirectory(examples)
endif()
