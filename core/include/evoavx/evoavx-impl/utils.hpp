#pragma once
#include "global.hpp"

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
}
