#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"

namespace evo::c
{
	template <cc::static_settings S>
	class BlendAlpha
	{
	private:
		f64_t m_alpha = 0.2;

	public:
		constexpr static bool s_twins = true;
		constexpr static bool s_fusedXM = true;
		constexpr static bool s_forceDomain = true;

		void set_alpha(f64_t alpha) noexcept;
		f64_t get_alpha() const noexcept;

		void init_generation(Island<S>&) {}
		void init_wave(Island<S>&) {}

		template <typename M>
		void perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output);
	};
}

namespace evo::c
{
	template <cc::static_settings S>
	inline void BlendAlpha<S>::set_alpha(f64_t alpha) noexcept
	{
		m_alpha = alpha;
	}

	template <cc::static_settings S>
	inline f64_t BlendAlpha<S>::get_alpha() const noexcept
	{
		return m_alpha;
	}

	template <cc::static_settings S> template <typename M>
	inline void BlendAlpha<S>::perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output)
	{
		Random<S>& random = island.m_random;
		u64_t length = island.m_realGenomeLength;
		f64_t* output2 = output + length;
		__m512d alpha = _mm512_set1_pd(m_alpha);

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
			__m512d valA = _mm512_load_pd(chromA + i);
			__m512d valB = _mm512_load_pd(chromB + i);
			__m512d minR = _mm512_min_pd(valA, valB);
			__m512d maxR = _mm512_max_pd(valA, valB);
			__m512d diff = _mm512_sub_pd(maxR, minR);
			__m512d extd = _mm512_mul_pd(diff, alpha);

			minR = _mm512_sub_pd(minR, extd);
			maxR = _mm512_add_pd(maxR, extd);
			diff = _mm512_sub_pd(maxR, minR);
			valA = _mm512_fmadd_pd(random.next_512d(), diff, minR);
			valB = _mm512_fmadd_pd(random.next_512d(), diff, minR);

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
