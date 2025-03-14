#pragma once
#include "../global.hpp"
#include "../algorithm.hpp"
#include "../evaluator.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"
#include "../statistics.hpp"

namespace evo::s
{
	template <cc::static_settings S>
	class Tournament
	{
	private:
		u64_t m_size = 3;

		template <Extremum extremum>
		void implementation(Island<S>& island);

	public:
		void perform_selection(Island<S>& island);
		void set_tournament_size(u64_t size);
		u64_t get_tournament_size() const noexcept;
	};
}

namespace evo::s
{
	template <cc::static_settings S> template <Extremum extremum>
	inline void Tournament<S>::implementation(Island<S>& island)
	{
		f64_t* input = island.m_scores.get();
		u64_t* output = island.m_selected.get();
		u64_t inputSize = island.m_indivCount;
		u64_t outputSize = island.m_selIndivCount;

		__m512d bestVal;
		__m512i bestIdx = _mm512_setzero_si512();
		__m512i range = _mm512_set1_epi64(inputSize);

		for (u64_t i = 0; i < outputSize; i += g_vectorGenes)
		{
			if constexpr (extremum == Extremum::MAXIMUM)
				bestVal = _mm512_set1_pd(std::numeric_limits<f64_t>::lowest());
			else
				bestVal = _mm512_set1_pd(std::numeric_limits<f64_t>::max());

			for (u64_t j = 0; j < m_size; j++)
			{
				__m512i indices = island.m_random.range_512i(range);
				__m512d sample = _mm512_i64gather_pd(indices, input, 8);
				__mmask8 mask;

				if constexpr (extremum == Extremum::MAXIMUM)
					mask = _mm512_cmplt_pd_mask(bestVal, sample);
				else
					mask = _mm512_cmplt_pd_mask(sample, bestVal);

				bestVal = _mm512_mask_mov_pd(bestVal, mask, sample);
				bestIdx = _mm512_mask_mov_epi64(bestIdx, mask, indices);
			}

			_mm512_store_epi64(output, bestIdx);
			output += g_vectorGenes;
		}
	}

	template <cc::static_settings S>
	inline void Tournament<S>::perform_selection(Island<S>& island)
	{
		if (island.m_extremum == Extremum::MAXIMUM)
			implementation<Extremum::MAXIMUM>(island);
		else
			implementation<Extremum::MINIMUM>(island);
	}

	template <cc::static_settings S>
	inline void Tournament<S>::set_tournament_size(u64_t size)
	{
		if (size == 0)
			throw std::invalid_argument("Tournament size must be greater than zero");

		m_size = size;
	}

	template <cc::static_settings S>
	inline u64_t Tournament<S>::get_tournament_size() const noexcept
	{
		return m_size;
	}
}
