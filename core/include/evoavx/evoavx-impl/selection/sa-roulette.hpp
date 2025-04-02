#pragma once
#include "../global.hpp"
#include "../island-dec.hpp"
#include "../random/random.hpp"
#include "../statistics.hpp"

namespace evo::s
{
	template <cc::static_settings S>
	class SaRouletteWindow
	{
	public:
		constexpr static bool s_usesAux = false;

	private:
		f64_t m_offset = 0.1;

		template <Extremum extremum>
		void implementation(Island<S>& island);

	public:
		void perform(Island<S>& island);
		void set_offset(f64_t offset);
		f64_t get_offset() const noexcept;
	};

	template <cc::static_settings S>
	class SaRouletteSigma
	{
	public:
		constexpr static bool s_usesAux = false;

	private:
		f64_t m_offset = 1.0;
		f64_t m_minimum = 0.1;
		f64_t m_factor = 2.0;
		f64_t m_epsilon = 1e-12;

		template <Extremum extremum>
		void implementation(Island<S>& island);

	public:
		void perform(Island<S>& island);
		void set_parameters(f64_t offset, f64_t minimum, f64_t factor, f64_t epsilon);

		f64_t get_offset() const noexcept;
		f64_t get_minimum() const noexcept;
		f64_t get_factor() const noexcept;
		f64_t get_epsilon() const noexcept;
	};
}

namespace evo::s
{
	template <cc::static_settings S> template <Extremum extremum>
	inline void SaRouletteWindow<S>::implementation(Island<S>& island)
	{
		f64_t* input = island.m_scores.get();
		u64_t* output = island.m_selected.get();
		u64_t inputSize = island.m_indivCount;
		u64_t outputSize = island.m_selIndivCount;

		f64_t max = island.m_statistics.m_maximum;
		f64_t min = island.m_statistics.m_minimum;

		__m512i range = _mm512_set1_epi64(inputSize);
		__m512d scale = _mm512_set1_pd(max - min + m_offset);
		__m512d offset;

		if constexpr (extremum == Extremum::MAXIMUM)
			offset = _mm512_set1_pd(min - m_offset);
		else
			offset = _mm512_set1_pd(max + m_offset);

		u64_t counter = g_vectorGenes;
		while (counter < outputSize)
		{
			__m512i indices = island.m_random.range_512i(range);
			__m512d sample = _mm512_i64gather_pd(indices, input, 8);
			__m512d thresholds = _mm512_mul_pd(island.m_random.next_512d(), scale);
			__m512d weights;

			if constexpr (extremum == Extremum::MAXIMUM)
				weights = _mm512_sub_pd(sample, offset);
			else
				weights = _mm512_sub_pd(offset, sample);

			__mmask8 selected = _mm512_cmple_pd_mask(thresholds, weights);
			u64_t inc = static_cast<u64_t>(_mm_popcnt_u32(_cvtmask8_u32(selected)));
			_mm512_mask_compressstoreu_epi64(output, selected, indices);

			counter += inc;
			output += inc;
		}

		counter -= g_vectorGenes;
		while (counter < outputSize)
		{
			__m512i indices = island.m_random.range_512i(range);
			__m512d sample = _mm512_i64gather_pd(indices, input, 8);
			__m512d thresholds = _mm512_mul_pd(island.m_random.next_512d(), scale);
			__m512d weights;

			if constexpr (extremum == Extremum::MAXIMUM)
				weights = _mm512_sub_pd(sample, offset);
			else
				weights = _mm512_sub_pd(offset, sample);

			__mmask8 selected = _mm512_cmple_pd_mask(thresholds, weights);
			__m512i compressed = _mm512_maskz_compress_epi64(selected, indices);

			u64_t inc = static_cast<u64_t>(_mm_popcnt_u32(_cvtmask8_u32(selected)));
			u64_t writeBits = (1ULL << std::min(inc, outputSize - counter)) - 1ULL;
			_mm512_mask_storeu_epi64(output, _cvtu32_mask8(writeBits), compressed);

			counter += inc;
			output += inc;
		}
	}

	template <cc::static_settings S>
	inline void SaRouletteWindow<S>::perform(Island<S>& island)
	{
		if (island.m_extremum == Extremum::MAXIMUM)
			implementation<Extremum::MAXIMUM>(island);
		else
			implementation<Extremum::MINIMUM>(island);
	}

	template <cc::static_settings S>
	inline void SaRouletteWindow<S>::set_offset(f64_t offset)
	{
		if (offset < 0.0)
			throw std::invalid_argument("Offset value can't be negative");

		m_offset = offset;
	}

	template <cc::static_settings S>
	inline f64_t SaRouletteWindow<S>::get_offset() const noexcept
	{
		return m_offset;
	}

	template <cc::static_settings S> template <Extremum extremum>
	inline void SaRouletteSigma<S>::implementation(Island<S>& island)
	{
		f64_t* input = island.m_scores.get();
		u64_t* output = island.m_selected.get();
		u64_t inputSize = island.m_indivCount;
		u64_t outputSize = island.m_selIndivCount;

		__m512i range = _mm512_set1_epi64(inputSize);
		if (island.m_statistics.m_stdev < m_epsilon)
		{
			for (u64_t i = 0; i < outputSize; i += g_vectorGenes)
				_mm512_store_epi64(output + i, island.m_random.range_512i(range));

			return;
		}

		f64_t coefficient = 1.0 / (island.m_statistics.m_stdev * m_factor);
		f64_t maxDifference;

		if constexpr (extremum == Extremum::MAXIMUM)
			maxDifference = island.m_statistics.m_maximum - island.m_statistics.m_mean;
		else
			maxDifference = island.m_statistics.m_mean - island.m_statistics.m_minimum;

		__m512d mean = _mm512_set1_pd(island.m_statistics.m_mean);
		__m512d scale = _mm512_set1_pd(maxDifference * coefficient + m_offset);
		__m512d factor = _mm512_set1_pd(coefficient);
		__m512d offset = _mm512_set1_pd(m_offset);
		__m512d minimum = _mm512_set1_pd(m_minimum);

		u64_t counter = g_vectorGenes;
		while (counter < outputSize)
		{
			__m512i indices = island.m_random.range_512i(range);
			__m512d sample = _mm512_i64gather_pd(indices, input, 8);
			__m512d thresholds = _mm512_mul_pd(island.m_random.next_512d(), scale);
			__m512d difference;

			if constexpr (extremum == Extremum::MAXIMUM)
				difference = _mm512_sub_pd(sample, mean);
			else
				difference = _mm512_sub_pd(mean, sample);

			__m512d weights = _mm512_fmadd_pd(difference, factor, offset);
			weights = _mm512_max_pd(weights, minimum);

			__mmask8 selected = _mm512_cmple_pd_mask(thresholds, weights);
			u64_t inc = static_cast<u64_t>(_mm_popcnt_u32(_cvtmask8_u32(selected)));
			_mm512_mask_compressstoreu_epi64(output, selected, indices);

			counter += inc;
			output += inc;
		}

		counter -= g_vectorGenes;
		while (counter < outputSize)
		{
			__m512i indices = island.m_random.range_512i(range);
			__m512d sample = _mm512_i64gather_pd(indices, input, 8);
			__m512d thresholds = _mm512_mul_pd(island.m_random.next_512d(), scale);
			__m512d difference;

			if constexpr (extremum == Extremum::MAXIMUM)
				difference = _mm512_sub_pd(sample, mean);
			else
				difference = _mm512_sub_pd(mean, sample);

			__m512d weights = _mm512_fmadd_pd(difference, factor, offset);
			weights = _mm512_max_pd(weights, minimum);

			__mmask8 selected = _mm512_cmple_pd_mask(thresholds, weights);
			__m512i compressed = _mm512_maskz_compress_epi64(selected, indices);

			u64_t inc = static_cast<u64_t>(_mm_popcnt_u32(_cvtmask8_u32(selected)));
			u64_t writeBits = (1ULL << std::min(inc, outputSize - counter)) - 1ULL;
			_mm512_mask_storeu_epi64(output, _cvtu32_mask8(writeBits), compressed);

			counter += inc;
			output += inc;
		}
	}

	template <cc::static_settings S>
	inline void SaRouletteSigma<S>::perform(Island<S>& island)
	{
		if (island.m_extremum == Extremum::MAXIMUM)
			implementation<Extremum::MAXIMUM>(island);
		else
			implementation<Extremum::MINIMUM>(island);
	}

	template <cc::static_settings S>
	inline void SaRouletteSigma<S>::set_parameters(f64_t offset, f64_t minimum, f64_t factor, f64_t epsilon)
	{
		if (offset <= 0.0 || minimum <= 0.0 || factor <= 0.0 || epsilon <= 0.0)
			throw std::invalid_argument("Arguments must be greater than zero");

		m_offset = offset;
		m_minimum = minimum;
		m_factor = factor;
		m_epsilon = epsilon;
	}

	template <cc::static_settings S>
	inline f64_t SaRouletteSigma<S>::get_offset() const noexcept
	{
		return m_offset;
	}

	template <cc::static_settings S>
	inline f64_t SaRouletteSigma<S>::get_minimum() const noexcept
	{
		return m_minimum;
	}

	template <cc::static_settings S>
	inline f64_t SaRouletteSigma<S>::get_factor() const noexcept
	{
		return m_factor;
	}

	template <cc::static_settings S>
	inline f64_t SaRouletteSigma<S>::get_epsilon() const noexcept
	{
		return m_epsilon;
	}
}
