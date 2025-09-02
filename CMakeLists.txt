cmake_minimum_required(VERSION 3.25)
project(evoavx-root LANGUAGES CXX)

option(EVOAVX_BUILD_TESTS "Build unit tests for EvoAVX" ${PROJECT_IS_TOP_LEVEL})
option(EVOAVX_BUILD_EXAMPLES "Build examples for EvoAVX" ${PROJECT_IS_TOP_LEVEL})

add_subdirectory(core)

if(${EVOAVX_BUILD_TESTS})
	add_subdirectory(tests)
endif()

if(${EVOAVX_BUILD_EXAMPLES})
	file(GLOB_RECURSE EVOAVX_EXAMPLES_CPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/examples/*.cpp")
	foreach(CPP_FILE IN LISTS EVOAVX_EXAMPLES_CPP)
		cmake_path(GET CPP_FILE STEM CPP_FILENAME)
		set(TARGET_NAME "evoavx-${CPP_FILENAME}")
		add_executable("${TARGET_NAME}" "${CPP_FILE}")
		target_link_libraries("${TARGET_NAME}" PRIVATE evoavx)
		set_target_properties("${TARGET_NAME}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
	endforeach()
endif()
