#pragma once
#include "../global.hpp"

namespace evo::cc
{
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
}

namespace evo
{
	template <cc::wide_prng_engine EngineT>
	class Random : public EngineT
	{
	public:
		__m512d next_512d() noexcept;
		__m512i range_512i_52(u64_t range) noexcept;
		__m512i range_512i_52(__m512i range) noexcept;
	};
}

namespace evo
{
	template <cc::wide_prng_engine EngineT>
	inline __m512d Random<EngineT>::next_512d() noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i shifted = _mm512_srli_epi64(source, 11);
		__m512d converted = _mm512_cvtepu64_pd(shifted);

		return _mm512_mul_pd(converted, _mm512_set1_pd(0x1.0p-53));
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::range_512i_52(u64_t range) noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i rangeVec = _mm512_set1_epi64(range);
		__m512i low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, rangeVec);
		__mmask8 mask = _mm512_cmplt_epu64_mask(low, rangeVec);

		if (mask) [[unlikely]]
		{
			__m512i t = _mm512_set1_epi64((1ULL << 52) % range);
			while (mask = _mm512_cmplt_epu64_mask(low, t))
			{
				source = EngineT::next_512i();
				low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, rangeVec);
			}
		}

		return _mm512_madd52hi_epu64(_mm512_setzero_si512(), source, rangeVec);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::range_512i_52(__m512i range) noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);
		__mmask8 mask = _mm512_cmplt_epu64_mask(low, range);

		if (mask) [[unlikely]]
		{
			u64_t rangeArray[8];
			_mm512_storeu_epi64(rangeArray, range);

			rangeArray[0] = (1ULL << 52) % rangeArray[0];
			rangeArray[1] = (1ULL << 52) % rangeArray[1];
			rangeArray[2] = (1ULL << 52) % rangeArray[2];
			rangeArray[3] = (1ULL << 52) % rangeArray[3];
			rangeArray[4] = (1ULL << 52) % rangeArray[4];
			rangeArray[5] = (1ULL << 52) % rangeArray[5];
			rangeArray[6] = (1ULL << 52) % rangeArray[6];
			rangeArray[7] = (1ULL << 52) % rangeArray[7];

			__m512i t = _mm512_loadu_epi64(rangeArray);
			while (mask = _mm512_cmplt_epu64_mask(low, t))
			{
				source = EngineT::next_512i();
				low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);
			}
		}

		return _mm512_madd52hi_epu64(_mm512_setzero_si512(), source, range);
	}
}
