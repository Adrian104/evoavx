#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"

namespace evo::c
{
	template <cc::static_settings S>
	class Uniform
	{
	private:
		f64_t m_swapProb = 0.5;

	public:
		constexpr static bool s_twins = true;
		constexpr static bool s_fusedXM = true;
		constexpr static bool s_forceDomain = false;

		void set_swap_probability(f64_t prob);
		f64_t get_swap_probability() const noexcept;

		void init_generation(Island<S>&) {}
		void init_wave(Island<S>&) {}

		template <typename M>
		void perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output);
	};
}

namespace evo::c
{
	template <cc::static_settings S>
	inline void Uniform<S>::set_swap_probability(f64_t prob)
	{
		if (prob < 0.0 || prob > 1.0)
			throw std::invalid_argument("Probability must be within [0, 1]");

		m_swapProb = prob;
	}

	template <cc::static_settings S>
	inline f64_t Uniform<S>::get_swap_probability() const noexcept
	{
		return m_swapProb;
	}

	template <cc::static_settings S> template <typename M>
	inline void Uniform<S>::perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output)
	{
		Random<S>& random = island.m_random;
		u64_t length = island.m_realGenomeLength;
		f64_t* output2 = output + length;
		__m512d swapProb = _mm512_set1_pd(m_swapProb);

		[[maybe_unused]] f64_t* minPtr;
		[[maybe_unused]] f64_t* maxPtr;
		[[maybe_unused]] f64_t prob;

		if constexpr (!std::same_as<M, void>)
		{
			minPtr = island.m_minDomain.get();
			maxPtr = island.m_maxDomain.get();
			prob = island.m_mutationProb;
		}

		for (u64_t i = 0; i < length; i += g_vectorGenes)
		{
			__m512d srcA = _mm512_load_pd(chromA + i);
			__m512d srcB = _mm512_load_pd(chromB + i);
			__mmask8 mask = _mm512_cmplt_pd_mask(random.next_512d(), swapProb);
			__m512d valA = _mm512_mask_blend_pd(mask, srcA, srcB);
			__m512d valB = _mm512_mask_blend_pd(mask, srcB, srcA);

			if constexpr (!std::same_as<M, void>)
			{
				__m512d min = _mm512_load_pd(minPtr + i);
				__m512d max = _mm512_load_pd(maxPtr + i);

				valA = M::perform(valA, min, max, prob, random);
				valB = M::perform(valB, min, max, prob, random);
			}

			_mm512_store_pd(output + i, valA);
			_mm512_store_pd(output2 + i, valB);
		}
	}
}
