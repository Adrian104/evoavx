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
#include <immintrin.h>
#include <windows.h>
#endif

#ifdef EVO_OS_LINUX
#include <immintrin.h>
#include <unistd.h>
#endif

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <new>
#include <type_traits>

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
}
