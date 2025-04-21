#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"

namespace evo::c
{
	template <cc::static_settings S>
	class Average
	{
	public:
		constexpr static bool s_twins = false;
		constexpr static bool s_fusedXM = true;
		constexpr static bool s_forceDomain = false;

		void init_generation(Island<S>&) {}
		void init_wave(Island<S>&) {}

		template <typename M>
		void perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output);
	};
}

namespace evo::c
{
	template <cc::static_settings S> template <typename M>
	inline void Average<S>::perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output)
	{
		[[maybe_unused]] f64_t* minPtr;
		[[maybe_unused]] f64_t* maxPtr;
		[[maybe_unused]] f64_t prob;

		if constexpr (!std::same_as<M, void>)
		{
			minPtr = island.m_minDomain.get();
			maxPtr = island.m_maxDomain.get();
			prob = island.m_mutationProb;
		}

		u64_t length = island.m_realGenomeLength;
		__m512d mul = _mm512_set1_pd(0.5);

		for (u64_t i = 0; i < length; i += g_vectorGenes)
		{
			__m512d valA = _mm512_load_pd(chromA + i);
			__m512d valB = _mm512_load_pd(chromB + i);
			__m512d sum = _mm512_add_pd(valA, valB);
			__m512d avg = _mm512_mul_pd(sum, mul);

			if constexpr (!std::same_as<M, void>)
			{
				__m512d min = _mm512_load_pd(minPtr + i);
				__m512d max = _mm512_load_pd(maxPtr + i);

				avg = M::perform(avg, min, max, prob, island.m_random);
			}

			_mm512_store_pd(output + i, avg);
		}
	}
}
