#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../shuffle.hpp"
#include "../random/random.hpp"

namespace evo::s
{
	template <cc::static_settings S>
	class UnbiasedTournament
	{
	private:
		unique<u64_t[]> m_indices;
		u64_t m_count = 0;

		template <Extremum extremum>
		void implementation(Island<S>& island);

	public:
		void perform(Island<S>& island);
	};
}

namespace evo::s
{
	template <cc::static_settings S> template <Extremum extremum>
	inline void UnbiasedTournament<S>::implementation(Island<S>& island)
	{
		u64_t* indices = m_indices.get();
		f64_t* scores = island.m_scores.get();
		u64_t* output = island.m_selected.get();
		u64_t floorCount = alignment_floor<u64_t>(m_count);

		__m512i idxA = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
		__m512i inc = _mm512_set1_epi64(g_vectorGenes);

		for (u64_t i = 0; i < floorCount; i += g_vectorGenes)
		{
			__m512i idxB = _mm512_load_epi64(indices + i);
			__m512d valA = _mm512_load_pd(scores + i);
			__m512d valB = _mm512_i64gather_pd(idxB, scores, 8);
			__mmask8 mask;

			if constexpr (extremum == Extremum::MAXIMUM)
				mask = _mm512_cmplt_pd_mask(valA, valB);
			else
				mask = _mm512_cmplt_pd_mask(valB, valA);

			__m512i winners = _mm512_mask_blend_epi64(mask, idxA, idxB);
			_mm512_store_epi64(output + i, winners);
			idxA = _mm512_add_epi64(idxA, inc);
		}

		for (u64_t a = floorCount; a < m_count; ++a)
		{
			const u64_t b = indices[a];
			if constexpr (extremum == Extremum::MAXIMUM)
				output[a] = scores[a] < scores[b] ? b : a;
			else
				output[a] = scores[b] < scores[a] ? b : a;
		}
	}

	template <cc::static_settings S>
	inline void UnbiasedTournament<S>::perform(Island<S>& island)
	{
		const u64_t count = island.m_indivCount;
		if (count != island.m_selIndivCount)
			throw std::runtime_error("The number of selected individuals must be equal to the total number of individuals");

		if (m_count != count)
		{
			m_count = count;
			m_indices = unique<u64_t[]>(allocate<u64_t>(alignment_ceil<u64_t>(count)));
		}

		u64_t* const ptr = m_indices.get();
		__m512i val = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
		__m512i inc = _mm512_set1_epi64(g_vectorGenes);

		for (u64_t i = 0; i < count; i += g_vectorGenes)
		{
			_mm512_store_epi64(ptr + i, val);
			val = _mm512_add_epi64(val, inc);
		}

		derange(m_indices.get(), count, island.m_random);
		if (island.m_extremum == Extremum::MAXIMUM)
			implementation<Extremum::MAXIMUM>(island);
		else
			implementation<Extremum::MINIMUM>(island);
	}
}
