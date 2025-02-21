#pragma once
#include "../global.hpp"
#include "../utils.hpp"

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
		__m512i range_512i(__m512i range) noexcept;
		__m512i range_512i(__m512i range, __m512i t) noexcept;
		__m512i range_512i_52(__m512i range) noexcept;
		__m512i range_512i_52(__m512i range, __m512i t) noexcept;

		static __m512i compute_t(u64_t range) noexcept;
		static __m512i compute_t(__m512i range) noexcept;
		static __m512i compute_t_52(u64_t range) noexcept;
		static __m512i compute_t_52(__m512i range) noexcept;
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
	inline __m512i Random<EngineT>::range_512i(__m512i range) noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i low = _mm512_mullo_epi64(source, range);

		if (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, range))) [[unlikely]]
		{
			__m512i t = compute_t(range);
			while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t)))
			{
				source = EngineT::next_512i();
				low = _mm512_mullo_epi64(source, range);
			}
		}

		return mulhi_512i64(source, range);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::range_512i(__m512i range, __m512i t) noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i low = _mm512_mullo_epi64(source, range);

		while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t))) [[unlikely]]
		{
			source = EngineT::next_512i();
			low = _mm512_mullo_epi64(source, range);
		}

		return mulhi_512i64(source, range);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::range_512i_52(__m512i range) noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);

		if (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, range))) [[unlikely]]
		{
			__m512i t = compute_t_52(range);
			while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t)))
			{
				source = EngineT::next_512i();
				low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);
			}
		}

		return _mm512_madd52hi_epu64(_mm512_setzero_si512(), source, range);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::range_512i_52(__m512i range, __m512i t) noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);

		while (_cvtmask8_u32(_mm512_cmplt_epu64_mask(low, t))) [[unlikely]]
		{
			source = EngineT::next_512i();
			low = _mm512_madd52lo_epu64(_mm512_setzero_si512(), source, range);
		}

		return _mm512_madd52hi_epu64(_mm512_setzero_si512(), source, range);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::compute_t(u64_t range) noexcept
	{
		return _mm512_set1_epi64(-range % range);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::compute_t(__m512i range) noexcept
	{
		u64_t temp[8];
		_mm512_storeu_epi64(temp, range);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4146)
#endif

		temp[0] = -temp[0] % temp[0];
		temp[1] = -temp[1] % temp[1];
		temp[2] = -temp[2] % temp[2];
		temp[3] = -temp[3] % temp[3];
		temp[4] = -temp[4] % temp[4];
		temp[5] = -temp[5] % temp[5];
		temp[6] = -temp[6] % temp[6];
		temp[7] = -temp[7] % temp[7];

#ifdef _MSC_VER
#pragma warning(pop)
#endif

		return _mm512_loadu_epi64(temp);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::compute_t_52(u64_t range) noexcept
	{
		return _mm512_set1_epi64((1ULL << 52) % range);
	}

	template <cc::wide_prng_engine EngineT>
	inline __m512i Random<EngineT>::compute_t_52(__m512i range) noexcept
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
