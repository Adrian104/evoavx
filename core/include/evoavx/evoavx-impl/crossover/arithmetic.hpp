#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"

namespace evo::c
{
	template <cc::static_settings S>
	class Arithmetic
	{
	private:
		u64_t m_index = 0;
		f64_t m_alpha[g_vectorGenes]{};

	public:
		constexpr static bool s_twins = true;
		constexpr static bool s_fusedXM = true;
		constexpr static bool s_forceDomain = false;

		void init_generation(Island<S>&) {}
		void init_wave(Island<S>& island);

		template <typename M>
		void perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output);
	};
}

namespace evo::c
{
	template <cc::static_settings S>
	inline void Arithmetic<S>::init_wave(Island<S>& island)
	{
		m_index = 0;
		_mm512_storeu_pd(m_alpha, island.m_random.next_512d());
	}

	template <cc::static_settings S> template <typename M>
	inline void Arithmetic<S>::perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output)
	{
		u64_t length = island.m_realGenomeLength;
		f64_t* output2 = output + length;

		[[maybe_unused]] f64_t* minPtr;
		[[maybe_unused]] f64_t* maxPtr;
		[[maybe_unused]] f64_t prob;

		if constexpr (!std::same_as<M, void>)
		{
			minPtr = island.m_minDomain.get();
			maxPtr = island.m_maxDomain.get();
			prob = island.m_mutationProb;
		}

		__m512d alpha = _mm512_set1_pd(m_alpha[m_index]);
		__m512d beta = _mm512_set1_pd(1.0 - m_alpha[m_index++]);

		for (u64_t i = 0; i < length; i += g_vectorGenes)
		{
			__m512d valA = _mm512_load_pd(chromA + i);
			__m512d valB = _mm512_load_pd(chromB + i);
			__m512d part1 = _mm512_mul_pd(beta, valB);
			__m512d part2 = _mm512_mul_pd(beta, valA);

			valA = _mm512_fmadd_pd(alpha, valA, part1);
			valB = _mm512_fmadd_pd(alpha, valB, part2);

			if constexpr (!std::same_as<M, void>)
			{
				__m512d min = _mm512_load_pd(minPtr + i);
				__m512d max = _mm512_load_pd(maxPtr + i);

				valA = M::perform(valA, min, max, prob, island.m_random);
				valB = M::perform(valB, min, max, prob, island.m_random);
			}

			_mm512_store_pd(output + i, valA);
			_mm512_store_pd(output2 + i, valB);
		}
	}
}
