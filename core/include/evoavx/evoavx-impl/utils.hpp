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

	template <typename T>
	inline T* allocate(std::size_t count)
	{
		constexpr static std::size_t s_cacheLineSize = std::hardware_constructive_interference_size;
		constexpr static std::size_t s_vectorSize = 512 / 8;
		constexpr static std::size_t s_alignment = s_cacheLineSize > s_vectorSize ? s_cacheLineSize : s_vectorSize;

		std::size_t rem = count % s_alignment;
		if (rem)
			count += s_alignment - rem;

#ifdef EVO_OS_WINDOWS
		T* ptr = static_cast<T*>(_aligned_malloc(sizeof(T) * count, s_alignment));
#else
		T* ptr = static_cast<T*>(std::aligned_alloc(s_alignment, sizeof(T) * count));
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

	template <typename T>
	class Deleter
	{
	public:
		void operator()(T* ptr) const { deallocate(ptr); }
	};
}
