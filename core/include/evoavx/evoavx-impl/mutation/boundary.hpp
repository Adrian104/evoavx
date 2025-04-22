#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"

namespace evo::m
{
	template <cc::static_settings S, bool forceDomain>
	class Boundary
	{
	public:
		constexpr static bool s_fusedXM = true;

		void init_generation(Island<S>&) {}
		void init_wave(Island<S>&) {}

		void perform(Island<S>& island, f64_t* chrom);
		static __m512d perform(__m512d values, __m512d min, __m512d max, f64_t prob, Random<S>& random);
	};
}

namespace evo::m
{
	template <cc::static_settings S, bool forceDomain>
	inline void Boundary<S, forceDomain>::perform(Island<S>& island, f64_t* chrom)
	{
		const f64_t* const minPtr = island.m_minDomain.get();
		const f64_t* const maxPtr = island.m_maxDomain.get();
		const u64_t length = island.m_realGenomeLength;

		Random<S>& random = island.m_random;
		__m512d prob = _mm512_set1_pd(island.m_mutationProb);

		for (u64_t i = 0; i < length; i += g_vectorGenes)
		{
			__m512d min = _mm512_load_pd(minPtr + i);
			__m512d max = _mm512_load_pd(maxPtr + i);
			__m512d values = _mm512_load_pd(chrom + i);
			__mmask8 flip = _mm512_movepi64_mask(random.next_512i());
			__mmask8 mask = _mm512_cmplt_pd_mask(random.next_512d(), prob);
			__m512d boundary = _mm512_mask_blend_pd(flip, min, max);
			__m512d result = _mm512_mask_blend_pd(mask, values, boundary);

			if constexpr (forceDomain)
			{
				result = _mm512_max_pd(result, min);
				result = _mm512_min_pd(result, max);
			}

			_mm512_store_pd(chrom + i, result);
		}
	}

	template <cc::static_settings S, bool forceDomain>
	inline __m512d Boundary<S, forceDomain>::perform(__m512d values, __m512d min, __m512d max, f64_t prob, Random<S>& random)
	{
		__mmask8 flip = _mm512_movepi64_mask(random.next_512i());
		__mmask8 mask = _mm512_cmplt_pd_mask(random.next_512d(), _mm512_set1_pd(prob));
		__m512d boundary = _mm512_mask_blend_pd(flip, min, max);
		__m512d result = _mm512_mask_blend_pd(mask, values, boundary);

		if constexpr (forceDomain)
		{
			result = _mm512_max_pd(result, min);
			result = _mm512_min_pd(result, max);
		}

		return result;
	}
}
