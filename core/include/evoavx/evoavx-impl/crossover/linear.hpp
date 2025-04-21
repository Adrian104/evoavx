#pragma once
#include "../global.hpp"
#include "../evaluator.hpp"
#include "../island-dec.hpp"

namespace evo::c
{
	template <cc::static_settings S>
	class Linear
	{
	private:
		unique<f64_t[]> m_temp;
		u64_t m_length = 0;

	public:
		constexpr static bool s_twins = true;
		constexpr static bool s_fusedXM = false;
		constexpr static bool s_forceDomain = true;

		void init_generation(Island<S>& island);
		void init_wave(Island<S>&) {}

		template <typename M>
		void perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output);
	};
}

namespace evo::c
{
	template <cc::static_settings S>
	inline void Linear<S>::init_generation(Island<S>& island)
	{
		if (m_length != island.m_realGenomeLength)
		{
			m_length = island.m_realGenomeLength;
			m_temp = unique<f64_t[]>(allocate<f64_t>(m_length));
		}
	}

	template <cc::static_settings S> template <typename M>
	inline void Linear<S>::perform(Island<S>& island, f64_t* chromA, f64_t* chromB, f64_t* output)
	{
		f64_t* output2 = output + m_length;
		f64_t* output3 = m_temp.get();

		__m512d c05 = _mm512_set1_pd(0.5);
		__m512d c15 = _mm512_set1_pd(1.5);

		for (u64_t i = 0; i < m_length; i += g_vectorGenes)
		{
			__m512d srcA = _mm512_load_pd(chromA + i);
			__m512d srcB = _mm512_load_pd(chromB + i);
			__m512d srcA05 = _mm512_mul_pd(srcA, c05);
			__m512d srcB05 = _mm512_mul_pd(srcB, c05);
			__m512d valA = _mm512_add_pd(srcA05, srcB05);
			__m512d valB = _mm512_fmsub_pd(srcA, c15, srcB05);
			__m512d valC = _mm512_fmsub_pd(srcB, c15, srcA05);

			_mm512_store_pd(output + i, valA);
			_mm512_store_pd(output2 + i, valB);
			_mm512_store_pd(output3 + i, valC);
		}

		EvaluatorBase<S>& eval = *island.m_evaluator.get_used();

		f64_t scoreA = eval.evaluate_individual(output);
		f64_t scoreB = eval.evaluate_individual(output2);
		f64_t scoreC = eval.evaluate_individual(output3);

		if (island.m_extremum == Extremum::MAXIMUM)
		{
			if (scoreA < scoreB && scoreA < scoreC)
				std::memcpy(output, output3, m_length * sizeof(f64_t));
			else if (scoreB < scoreA && scoreB < scoreC)
				std::memcpy(output2, output3, m_length * sizeof(f64_t));
		}
		else
		{
			if (scoreA > scoreB && scoreA > scoreC)
				std::memcpy(output, output3, m_length * sizeof(f64_t));
			else if (scoreB > scoreA && scoreB > scoreC)
				std::memcpy(output2, output3, m_length * sizeof(f64_t));
		}
	}
}
