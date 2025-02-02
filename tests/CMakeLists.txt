project(evoavx-tests LANGUAGES CXX)

file(GLOB_RECURSE EVOAVX_TESTS_CPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE EVOAVX_TESTS_HPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.hpp")

find_package(Catch2 3 REQUIRED)
add_executable(${PROJECT_NAME} ${EVOAVX_TESTS_CPP} ${EVOAVX_TESTS_HPP})

target_link_libraries(${PROJECT_NAME} PRIVATE evoavx Catch2::Catch2WithMain)
target_precompile_headers(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/src/pch.hpp")
