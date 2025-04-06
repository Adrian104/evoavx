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

#if !defined(EVO_COMPILER_MSVC) && defined(_MSC_VER)
#define EVO_COMPILER_MSVC
#endif

#if !defined(EVO_COMPILER_CLANG) && defined(__clang__) && !defined(_MSC_VER)
#define EVO_COMPILER_CLANG
#endif

#if !defined(EVO_COMPILER_GNU) && defined(__GNUG__) && !defined(__clang__)
#define EVO_COMPILER_GNU
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
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
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
	using clk_t = std::chrono::high_resolution_clock;

	constexpr inline std::size_t g_vectorBits = 512;
	constexpr inline std::size_t g_vectorBytes = g_vectorBits / 8;
	constexpr inline std::size_t g_vectorGenes = g_vectorBytes / sizeof(f64_t);

	enum class Extremum
	{
		MINIMUM,
		MAXIMUM
	};

	enum class RangeAlg
	{
		LEMIRE_52,
		LEMIRE_64,
		LEMIRE_52_UNBIASED,
		LEMIRE_64_UNBIASED
	};

	enum class FusedXM
	{
		AUTO,
		DISABLED
	};

	enum class ForceDomain
	{
		AUTO,
		ENABLED
	};
}

namespace evo::cc
{
	template <typename DerivedT, typename BaseT>
	concept inherits_from = std::derived_from<DerivedT, BaseT> && !std::same_as<DerivedT, BaseT>;

	template <typename T>
	concept vector_element = (std::integral<T> || std::floating_point<T>) && std::has_single_bit(sizeof(T)) && (sizeof(T) < g_vectorBytes);

	template <typename EngineT>
	concept basic_prng_engine = std::is_nothrow_constructible_v<EngineT, u64_t>
		&& std::semiregular<EngineT> && requires(EngineT engine, u64_t seed)
	{
		{ engine.init(seed) } noexcept;
		{ engine.step() } noexcept;
		{ engine.next() } noexcept -> std::same_as<u64_t>;
	};

	template <typename EngineT>
	concept wide_prng_engine = std::is_nothrow_constructible_v<EngineT, u64_t>
		&& std::semiregular<EngineT> && requires(EngineT engine, u64_t seed)
	{
		{ engine.init(seed) } noexcept;
		{ engine.step() } noexcept;
		{ engine.jump() } noexcept;
		{ engine.next_512i() } noexcept -> std::same_as<__m512i>;
	};

	template <typename T>
	concept fitness_function = requires(const T cinstance, T instance, const f64_t* genes)
	{
		{ cinstance.length() } -> std::same_as<u64_t>;
		{ instance.evaluate(genes) } -> std::same_as<f64_t>;
	};

	template <typename S>
	concept static_settings = requires
	{
		requires cc::wide_prng_engine<typename S::prng_engine_t>;
		{ S::range_alg_v } -> std::convertible_to<RangeAlg>;
		{ S::fused_xm_v } -> std::convertible_to<FusedXM>;
		{ S::force_domain_v } -> std::convertible_to<ForceDomain>;
	};
}

namespace evo
{
	inline __m512i mulhi_512i64(__m512i a, __m512i b) noexcept
	{
		__m512i h0 = _mm512_srli_epi64(a, 32);
		__m512i h1 = _mm512_srli_epi64(b, 32);

		__m512i ll = _mm512_mul_epu32(a, b);
		__m512i lh = _mm512_mul_epu32(a, h1);
		__m512i hl = _mm512_mul_epu32(h0, b);
		__m512i hh = _mm512_mul_epu32(h0, h1);

		__m512i llh = _mm512_srli_epi64(ll, 32);
		__m512i hll = _mm512_and_epi64(hl, _mm512_set1_epi64(0xFFFFFFFF));

		lh = _mm512_add_epi64(lh, llh);
		lh = _mm512_add_epi64(lh, hll);

		__m512i hlh = _mm512_srli_epi64(hl, 32);
		__m512i lhh = _mm512_srli_epi64(lh, 32);

		hh = _mm512_add_epi64(hh, hlh);
		hh = _mm512_add_epi64(hh, lhh);

		return hh;
	}

	template <cc::vector_element T>
	inline u64_t alignment_floor(u64_t count) noexcept
	{
		constexpr static u64_t multiples = g_vectorBytes / sizeof(T);
		constexpr static u64_t mask = ~(multiples - 1);

		return count & mask;
	}

	template <cc::vector_element T>
	inline u64_t alignment_ceil(u64_t count) noexcept
	{
		constexpr static u64_t multiples = g_vectorBytes / sizeof(T);
		constexpr static u64_t mask = ~(multiples - 1);

		const u64_t floor = count & mask;
		const u64_t values[2]{ floor, floor + multiples };

		return values[count != floor];
	}

	template <cc::vector_element T>
	inline T* allocate(u64_t count)
	{
		const std::size_t bytes = count * sizeof(T);
		assert((bytes % g_vectorBytes) == 0);

#ifdef EVO_OS_WINDOWS
		T* ptr = static_cast<T*>(_aligned_malloc(bytes, g_vectorBytes));
#else
		T* ptr = static_cast<T*>(std::aligned_alloc(g_vectorBytes, bytes));
#endif

		assert(ptr != nullptr);
		return ptr;
	}

	inline void deallocate(void* ptr)
	{
		assert(ptr != nullptr);

#ifdef EVO_OS_WINDOWS
		_aligned_free(ptr);
#else
		std::free(ptr);
#endif
	}

	class Deleter
	{
	public:
		template <typename T>
		void operator()(T* ptr) const { deallocate(ptr); }
	};

	template <typename T>
	using unique = std::unique_ptr<T, Deleter>;

	template <cc::wide_prng_engine PrngEngineT, RangeAlg rangeAlg, FusedXM fusedXM, ForceDomain forceDomain>
	class StaticSettings
	{
	public:
		using prng_engine_t = PrngEngineT;
		constexpr static RangeAlg range_alg_v = rangeAlg;
		constexpr static FusedXM fused_xm_v = fusedXM;
		constexpr static ForceDomain force_domain_v = forceDomain;
	};
}
