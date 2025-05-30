#pragma once
#include "global.hpp"
#include "island-dec.hpp"

namespace evo
{
	template <cc::static_settings S>
	class Statistics
	{
	public:
		f64_t m_mean = 0;
		f64_t m_stdev = 0;
		f64_t m_minimum = 0;
		f64_t m_maximum = 0;
		u64_t m_minimumPos = 0;
		u64_t m_maximumPos = 0;
		u64_t m_generation = 0;
		u64_t m_stagnation = 0;

		void start();
		void refresh(Island<S>& island);
		void update(Island<S>& island, f64_t meanEstimate);
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline void Statistics<S>::start()
	{
		m_generation = 0;
		m_stagnation = 0;
	}

	template <cc::static_settings S>
	inline void Statistics<S>::refresh(Island<S>& island)
	{
		const u64_t count = alignment_floor<f64_t>(island.m_indivCount);
		const u32_t extra = (1 << (island.m_indivCount - count)) - 1;
		const f64_t* const scores = island.m_scores.get();
		const f64_t* const last = scores + count;

		__m512d sum0 = _mm512_setzero_pd();
		for (const f64_t* ptr = scores; ptr != last; ptr += g_vectorGenes)
			sum0 = _mm512_add_pd(sum0, _mm512_load_pd(ptr));

		if (extra)
		{
			__mmask8 mask = _cvtu32_mask8(extra);
			__m512d data = _mm512_maskz_load_pd(mask, last);
			sum0 = _mm512_mask_add_pd(sum0, mask, sum0, data);
		}

		const f64_t meanEstimate = _mm512_reduce_add_pd(sum0) / island.m_indivCount;

		__m512d sum1 = _mm512_setzero_pd();
		__m512d sum2 = _mm512_setzero_pd();
		__m512d mean = _mm512_set1_pd(meanEstimate);
		__m512d min = _mm512_set1_pd(std::numeric_limits<f64_t>::max());
		__m512d max = _mm512_set1_pd(std::numeric_limits<f64_t>::lowest());
		__m512i minPos = _mm512_setzero_si512();
		__m512i maxPos = _mm512_setzero_si512();
		__m512i crrPos = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
		__m512i inc = _mm512_set1_epi64(g_vectorGenes);

		for (const f64_t* ptr = scores; ptr != last; ptr += g_vectorGenes)
		{
			__m512d data = _mm512_load_pd(ptr);
			__m512d diff = _mm512_sub_pd(data, mean);
			__mmask8 minMask = _mm512_cmplt_pd_mask(data, min);
			__mmask8 maxMask = _mm512_cmplt_pd_mask(max, data);

			sum1 = _mm512_add_pd(diff, sum1);
			sum2 = _mm512_fmadd_pd(diff, diff, sum2);
			min = _mm512_mask_mov_pd(min, minMask, data);
			max = _mm512_mask_mov_pd(max, maxMask, data);
			minPos = _mm512_mask_mov_epi64(minPos, minMask, crrPos);
			maxPos = _mm512_mask_mov_epi64(maxPos, maxMask, crrPos);
			crrPos = _mm512_add_epi64(crrPos, inc);
		}

		if (extra)
		{
			__mmask8 mask = _cvtu32_mask8(extra);
			__m512d data = _mm512_maskz_load_pd(mask, last);
			__m512d diff1 = _mm512_sub_pd(data, mean);
			__m512d diff2 = _mm512_mul_pd(diff1, diff1);
			__mmask8 minMask = _mm512_cmplt_pd_mask(data, min);
			__mmask8 maxMask = _mm512_cmplt_pd_mask(max, data);

			minMask = _kand_mask8(minMask, mask);
			maxMask = _kand_mask8(maxMask, mask);

			sum1 = _mm512_mask_add_pd(sum1, mask, sum1, diff1);
			sum2 = _mm512_mask_add_pd(sum2, mask, sum2, diff2);
			min = _mm512_mask_mov_pd(min, minMask, data);
			max = _mm512_mask_mov_pd(max, maxMask, data);
			minPos = _mm512_mask_mov_epi64(minPos, minMask, crrPos);
			maxPos = _mm512_mask_mov_epi64(maxPos, maxMask, crrPos);
		}

		m_minimum = _mm512_reduce_min_pd(min);
		m_maximum = _mm512_reduce_max_pd(max);

		alignas(g_vectorBytes) u64_t minPosArray[g_vectorGenes];
		alignas(g_vectorBytes) u64_t maxPosArray[g_vectorGenes];

		_mm512_store_epi64(minPosArray, minPos);
		_mm512_store_epi64(maxPosArray, maxPos);

		__m512d hmin512 = _mm512_set1_pd(m_minimum);
		__m512d hmax512 = _mm512_set1_pd(m_maximum);
		u32_t minBits = _cvtmask8_u32(_mm512_cmpeq_pd_mask(min, hmin512));
		u32_t maxBits = _cvtmask8_u32(_mm512_cmpeq_pd_mask(max, hmax512));

		m_minimumPos = minPosArray[_tzcnt_u32(minBits)];
		m_maximumPos = maxPosArray[_tzcnt_u32(maxBits)];

		f64_t hsum1 = _mm512_reduce_add_pd(sum1);
		f64_t hsum2 = _mm512_reduce_add_pd(sum2);
		f64_t k = static_cast<f64_t>(island.m_indivCount);

		m_mean = meanEstimate + hsum1 / k;
		m_stdev = std::sqrt(hsum2 / k - (hsum1 * hsum1) / (k * k));
	}

	template <cc::static_settings S>
	inline void Statistics<S>::update(Island<S>& island, f64_t meanEstimate)
	{
		const u64_t count = alignment_floor<f64_t>(island.m_indivCount);
		const u32_t extra = (1 << (island.m_indivCount - count)) - 1;
		const f64_t* const scores = island.m_scores.get();
		const f64_t* const last = scores + count;

		__m512d sum1 = _mm512_setzero_pd();
		__m512d sum2 = _mm512_setzero_pd();
		__m512d mean = _mm512_set1_pd(meanEstimate);
		__m512d min = _mm512_set1_pd(std::numeric_limits<f64_t>::max());
		__m512d max = _mm512_set1_pd(std::numeric_limits<f64_t>::lowest());
		__m512i minPos = _mm512_setzero_si512();
		__m512i maxPos = _mm512_setzero_si512();
		__m512i crrPos = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
		__m512i inc = _mm512_set1_epi64(g_vectorGenes);

		for (const f64_t* ptr = scores; ptr != last; ptr += g_vectorGenes)
		{
			__m512d data = _mm512_load_pd(ptr);
			__m512d diff = _mm512_sub_pd(data, mean);
			__mmask8 minMask = _mm512_cmplt_pd_mask(data, min);
			__mmask8 maxMask = _mm512_cmplt_pd_mask(max, data);

			sum1 = _mm512_add_pd(diff, sum1);
			sum2 = _mm512_fmadd_pd(diff, diff, sum2);
			min = _mm512_mask_mov_pd(min, minMask, data);
			max = _mm512_mask_mov_pd(max, maxMask, data);
			minPos = _mm512_mask_mov_epi64(minPos, minMask, crrPos);
			maxPos = _mm512_mask_mov_epi64(maxPos, maxMask, crrPos);
			crrPos = _mm512_add_epi64(crrPos, inc);
		}

		if (extra)
		{
			__mmask8 mask = _cvtu32_mask8(extra);
			__m512d data = _mm512_maskz_load_pd(mask, last);
			__m512d diff1 = _mm512_sub_pd(data, mean);
			__m512d diff2 = _mm512_mul_pd(diff1, diff1);
			__mmask8 minMask = _mm512_cmplt_pd_mask(data, min);
			__mmask8 maxMask = _mm512_cmplt_pd_mask(max, data);

			minMask = _kand_mask8(minMask, mask);
			maxMask = _kand_mask8(maxMask, mask);

			sum1 = _mm512_mask_add_pd(sum1, mask, sum1, diff1);
			sum2 = _mm512_mask_add_pd(sum2, mask, sum2, diff2);
			min = _mm512_mask_mov_pd(min, minMask, data);
			max = _mm512_mask_mov_pd(max, maxMask, data);
			minPos = _mm512_mask_mov_epi64(minPos, minMask, crrPos);
			maxPos = _mm512_mask_mov_epi64(maxPos, maxMask, crrPos);
		}

		f64_t hmin = _mm512_reduce_min_pd(min);
		f64_t hmax = _mm512_reduce_max_pd(max);

		++m_generation;
		++m_stagnation;

		if ((island.m_extremum == Extremum::MINIMUM && hmin < m_minimum) ||
			(island.m_extremum == Extremum::MAXIMUM && hmax > m_maximum))
			m_stagnation = 0;

		m_minimum = hmin;
		m_maximum = hmax;

		alignas(g_vectorBytes) u64_t minPosArray[g_vectorGenes];
		alignas(g_vectorBytes) u64_t maxPosArray[g_vectorGenes];

		_mm512_store_epi64(minPosArray, minPos);
		_mm512_store_epi64(maxPosArray, maxPos);

		__m512d hmin512 = _mm512_set1_pd(hmin);
		__m512d hmax512 = _mm512_set1_pd(hmax);
		u32_t minBits = _cvtmask8_u32(_mm512_cmpeq_pd_mask(min, hmin512));
		u32_t maxBits = _cvtmask8_u32(_mm512_cmpeq_pd_mask(max, hmax512));

		m_minimumPos = minPosArray[_tzcnt_u32(minBits)];
		m_maximumPos = maxPosArray[_tzcnt_u32(maxBits)];

		f64_t hsum1 = _mm512_reduce_add_pd(sum1);
		f64_t hsum2 = _mm512_reduce_add_pd(sum2);
		f64_t k = static_cast<f64_t>(island.m_indivCount);

		m_mean = meanEstimate + hsum1 / k;
		m_stdev = std::sqrt(hsum2 / k - (hsum1 * hsum1) / (k * k));
	}
}
