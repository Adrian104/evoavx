#pragma once

#if !defined(EVO_OS_WINDOWS) && (defined(_WIN32) || defined(_WIN64))
#define EVO_OS_WINDOWS
#endif

#if !defined(EVO_OS_LINUX) && defined(__linux__)
#define EVO_OS_LINUX
#endif

#if defined(EVO_OS_WINDOWS) && defined(EVO_OS_LINUX)
#error "Invalid OS configuration"
#endif

#ifdef EVO_OS_WINDOWS
#define NOMINMAX
#include <immintrin.h>
#include <windows.h>
#endif

#ifdef EVO_OS_LINUX
#include <immintrin.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace evo
{
	using f32_t = float;
	using f64_t = double;
	using i8_t = std::int8_t;
	using i16_t = std::int16_t;
	using i32_t = std::int32_t;
	using i64_t = std::int64_t;
	using u8_t = std::uint8_t;
	using u16_t = std::uint16_t;
	using u32_t = std::uint32_t;
	using u64_t = std::uint64_t;

	constexpr inline std::size_t g_vectorBits = 512;
	constexpr inline std::size_t g_vectorBytes = g_vectorBits / 8;
	constexpr inline std::size_t g_vectorGenes = g_vectorBytes / sizeof(f64_t);
}

namespace evo::cc
{
	template <typename DerivedT, typename BaseT>
	concept inherits_from = std::derived_from<DerivedT, BaseT> && !std::same_as<DerivedT, BaseT>;

	template <typename T>
	concept vector_element = (std::integral<T> || std::floating_point<T>) && std::has_single_bit(sizeof(T)) && (sizeof(T) < g_vectorBytes);
}
