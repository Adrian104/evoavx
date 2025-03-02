#pragma once
#include "../global.hpp"

namespace evo
{
	template <cc::static_settings S>
	class Random : public S::prng_engine_t
	{
		using engine_t = typename S::prng_engine_t;
		constexpr static RangeAlg s_rangeAlg = S::range_alg_v;

	public:
		__m512d next_512d() noexcept;
		__m512i range_512i(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64);
		__m512i range_512i(__m512i range, __m512i t) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64);
		__m512i range_512i(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52);
		__m512i range_512i(__m512i range, __m512i t) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52);

		static __m512i compute_t(u64_t range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64);
		static __m512i compute_t(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64);
		static __m512i compute_t(u64_t range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52);
		static __m512i compute_t(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52);
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline __m512d Random<S>::next_512d() noexcept
	{
		__m512i source = engine_t::next_512i();
		__m512i shifted = _mm512_srli_epi64(source, 11);
		__m512d converted = _mm512_cvtepu64_pd(shifted);

		return _mm512_mul_pd(converted, _mm512_set1_pd(0x1.0p-53));
	}

	template <cc::static_settings S>
	inline __m512i Random<S>::range_512i(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64)
	{
		__m512i source = engine_t::next_512i();
		__m512i low = _mm512_mullo_epi64(source, range);

		if (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, range))) [[unlikely]]
		{
			__m512i t = compute_t(range);
			while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t)))
			{
				source = engine_t::next_512i();
				low = _mm512_mullo_epi64(source, range);
			}
		}

		return mulhi_512i64(source, range);
	}

	template <cc::static_settings S>
	inline __m512i Random<S>::range_512i(__m512i range, __m512i t) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64)
	{
		__m512i source = engine_t::next_512i();
		__m512i low = _mm512_mullo_epi64(source, range);

		while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t))) [[unlikely]]
		{
			source = engine_t::next_512i();
			low = _mm512_mullo_epi64(source, range);
		}

		return mulhi_512i64(source, range);
	}

	template <cc::static_settings S>
	inline __m512i Random<S>::range_512i(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52)
	{
		__m512i source = engine_t::next_512i();
		__m512i low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);

		if (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, range))) [[unlikely]]
		{
			__m512i t = compute_t(range);
			while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t)))
			{
				source = engine_t::next_512i();
				low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);
			}
		}

		return _mm512_madd52hi_epu64(_mm512_setzero_si512(), source, range);
	}

	template <cc::static_settings S>
	inline __m512i Random<S>::range_512i(__m512i range, __m512i t) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52)
	{
		__m512i source = engine_t::next_512i();
		__m512i low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);

		while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t))) [[unlikely]]
		{
			source = engine_t::next_512i();
			low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);
		}

		return _mm512_madd52hi_epu64(_mm512_setzero_si512(), source, range);
	}

#ifdef EVO_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable: 4146)
#endif

	template <cc::static_settings S>
	inline __m512i Random<S>::compute_t(u64_t range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64)
	{
		return _mm512_set1_epi64(-range % range);
	}

	template <cc::static_settings S>
	inline __m512i Random<S>::compute_t(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_64)
	{
		u64_t temp[8];
		_mm512_storeu_epi64(temp, range);

		temp[0] = -temp[0] % temp[0];
		temp[1] = -temp[1] % temp[1];
		temp[2] = -temp[2] % temp[2];
		temp[3] = -temp[3] % temp[3];
		temp[4] = -temp[4] % temp[4];
		temp[5] = -temp[5] % temp[5];
		temp[6] = -temp[6] % temp[6];
		temp[7] = -temp[7] % temp[7];

		return _mm512_loadu_epi64(temp);
	}

#ifdef EVO_COMPILER_MSVC
#pragma warning(pop)
#endif

	template <cc::static_settings S>
	inline __m512i Random<S>::compute_t(u64_t range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52)
	{
		return _mm512_set1_epi64((1ULL << 52) % range);
	}

	template <cc::static_settings S>
	inline __m512i Random<S>::compute_t(__m512i range) noexcept requires (s_rangeAlg == RangeAlg::LEMIRE_52)
	{
		u64_t temp[8];
		_mm512_storeu_epi64(temp, range);

		temp[0] = (1ULL << 52) % temp[0];
		temp[1] = (1ULL << 52) % temp[1];
		temp[2] = (1ULL << 52) % temp[2];
		temp[3] = (1ULL << 52) % temp[3];
		temp[4] = (1ULL << 52) % temp[4];
		temp[5] = (1ULL << 52) % temp[5];
		temp[6] = (1ULL << 52) % temp[6];
		temp[7] = (1ULL << 52) % temp[7];

		return _mm512_loadu_epi64(temp);
	}
}
