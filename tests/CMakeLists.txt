project(evoavx-tests LANGUAGES CXX)

file(GLOB_RECURSE EVOAVX_TESTS_CPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE EVOAVX_TESTS_HPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.hpp")

find_package(Catch2 3 QUIET)

if(Catch2_FOUND)
	message(STATUS "${PROJECT_NAME}: Found Catch2")
else()
	message(STATUS "${PROJECT_NAME}: Couldn't find Catch2 - downloading...")

	include(FetchContent)
	FetchContent_Declare(
		Catch2
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.15.1
		GIT_SHALLOW ON
	)

	FetchContent_MakeAvailable(Catch2)
endif()

add_executable(${PROJECT_NAME} ${EVOAVX_TESTS_CPP} ${EVOAVX_TESTS_HPP})

target_link_libraries(${PROJECT_NAME} PRIVATE evoavx Catch2::Catch2WithMain)
target_precompile_headers(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/src/pch.hpp")
