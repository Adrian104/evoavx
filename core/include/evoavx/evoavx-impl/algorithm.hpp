#pragma once
#include "global.hpp"
#include "island-dec.hpp"
#include "shuffle.hpp"
#include "random/random.hpp"

namespace evo::cc
{
	template <typename B, typename S>
	concept blueprint = requires(Island<S> island, f64_t f64, __m512d m512d, Random<S> rand,
		typename B::template selection_t<S> selection,
		typename B::template crossover_t<S> crossover,
		typename B::template mutation_t<S> mutation)
	{
		requires static_settings<S>;

		{ decltype(crossover)::s_twins } -> std::convertible_to<bool>;
		{ decltype(crossover)::s_fusedXM } -> std::convertible_to<bool>;
		{ decltype(crossover)::s_forceDomain } -> std::convertible_to<bool>;
		{ decltype(mutation)::s_fusedXM } -> std::convertible_to<bool>;

		selection.perform(island);
		crossover.template perform<decltype(mutation)>(island, &f64, &f64, &f64);
		mutation.perform(island, &f64);
		{ decltype(mutation)::perform(m512d, m512d, m512d, f64, rand) } -> std::same_as<__m512d>;

		crossover.init_generation(island);
		crossover.init_wave(island);
		mutation.init_generation(island);
		mutation.init_wave(island);

		{ B::s_picker } -> std::convertible_to<Picker>;
		{ B::s_elitism } -> std::convertible_to<Elitism>;
	};
}

namespace evo
{
	template <
		template <typename> typename SelectionT,
		template <typename> typename CrossoverT,
		template <typename, bool> typename MutationT,
		Picker picker = Picker::SHUFFLE,
		Elitism elitism = Elitism::ENABLED>
	class Blueprint
	{
	public:
		template <cc::static_settings S>
		using selection_t = SelectionT<S>;

		template <cc::static_settings S>
		using crossover_t = CrossoverT<S>;

		template <cc::static_settings S>
		using mutation_t = MutationT<S, crossover_t<S>::s_forceDomain
			|| S::force_domain_v == ForceDomain::ENABLED>;

		constexpr static Picker s_picker = picker;
		constexpr static Elitism s_elitism = elitism;
	};

	template <cc::static_settings S>
	class AlgorithmBase
	{
	public:
		virtual ~AlgorithmBase() = default;
		virtual void evolve(Island<S>& island) = 0;
	};

	template <cc::static_settings S, cc::blueprint<S> B>
	class Algorithm : public AlgorithmBase<S>
	{
	public:
		using selection_t = typename B::template selection_t<S>;
		using crossover_t = typename B::template crossover_t<S>;
		using mutation_t = typename B::template mutation_t<S>;

		constexpr static bool s_twins = crossover_t::s_twins;
		constexpr static bool s_fusedXM = crossover_t::s_fusedXM && mutation_t::s_fusedXM && S::fused_xm_v == FusedXM::AUTO;

		selection_t m_selection;
		crossover_t m_crossover;
		mutation_t m_mutation;

		void evolve(Island<S>& island) override;

	private:
		void perform_xm(Island<S>& island) requires (B::s_picker == Picker::RANDOM);
		void perform_xm(Island<S>& island) requires (B::s_picker == Picker::SHUFFLE);
		f64_t* xm_step(Island<S>& island, f64_t* a, f64_t* b, f64_t* out, bool crossover) requires (s_fusedXM);
		f64_t* xm_step(Island<S>& island, f64_t* a, f64_t* b, f64_t* out, bool crossover) requires (!s_fusedXM);
	};
}

namespace evo
{
	template <cc::static_settings S, cc::blueprint<S> B>
	inline void Algorithm<S, B>::evolve(Island<S>& island)
	{
		m_selection.perform(island);
		perform_xm(island);

		if constexpr (B::s_elitism == Elitism::ENABLED)
		{
			u64_t idx = island.m_extremum == Extremum::MINIMUM ?
				island.m_statistics.m_minimumPos : island.m_statistics.m_maximumPos;

			f64_t* src = island.m_current.get() + idx * island.m_realGenomeLength;
			f64_t* dest = island.m_next.get();
			u64_t bytes = island.m_realGenomeLength * sizeof(f64_t);

			std::memcpy(dest, src, bytes);
		}

		std::swap(island.m_current, island.m_next);
	}

	template <cc::static_settings S, cc::blueprint<S> B>
	inline void Algorithm<S, B>::perform_xm(Island<S>& island) requires (B::s_picker == Picker::RANDOM)
	{
		m_crossover.init_generation(island);
		m_mutation.init_generation(island);

		alignas(g_vectorBytes) u64_t offsetsA[8]{};
		alignas(g_vectorBytes) u64_t offsetsB[8]{};

		f64_t* src = island.m_current.get();
		f64_t* dest = island.m_next.get();
		u64_t* sel = island.m_selected.get();

		__m512i rangeA = _mm512_set1_epi64(island.m_selIndivCount);
		__m512i rangeB = _mm512_set1_epi64(island.m_selIndivCount - 1);
		__m512i scale = _mm512_set1_epi64(island.m_realGenomeLength);
		__m512d prob = _mm512_set1_pd(island.m_crossoverProb);

		u64_t count;
		if constexpr (s_twins)
			count = (island.m_indivCount + 1) >> 1;
		else
			count = island.m_indivCount;

		u64_t step = count & 0b111;
		if (step == 0)
			step = 8;

		while (count)
		{
			m_crossover.init_wave(island);
			m_mutation.init_wave(island);

			__m512i randA = island.m_random.range_512i(rangeA);
			__m512i randB = island.m_random.range_512i(rangeB);
			__m512d randX = island.m_random.next_512d();

			__mmask8 inc = _mm512_cmple_epu64_mask(randA, randB);
			u32_t cx = _cvtmask8_u32(_mm512_cmplt_pd_mask(randX, prob));

			randB = _mm512_mask_add_epi64(randB, inc, randB, _mm512_set1_epi64(1));

			__m512i indicesA = _mm512_i64gather_epi64(randA, sel, 8);
			__m512i indicesB = _mm512_i64gather_epi64(randB, sel, 8);

			_mm512_store_epi64(offsetsA, _mm512_mullo_epi64(indicesA, scale));
			_mm512_store_epi64(offsetsB, _mm512_mullo_epi64(indicesB, scale));

			for (u64_t i = 0; i < step; i++, cx >>= 1)
				dest = xm_step(island, src + offsetsA[i], src + offsetsB[i], dest, static_cast<bool>(cx & 1));

			count -= step;
			step = 8;
		}
	}

	template <cc::static_settings S, cc::blueprint<S> B>
	inline void Algorithm<S, B>::perform_xm(Island<S>& island) requires (B::s_picker == Picker::SHUFFLE)
	{
		m_crossover.init_generation(island);
		m_mutation.init_generation(island);

		__m512d prob = _mm512_set1_pd(island.m_crossoverProb);
		f64_t* src = island.m_current.get();
		f64_t* dest = island.m_next.get();
		u64_t scale = island.m_realGenomeLength;
		u64_t left;

		if constexpr (s_twins)
			left = (island.m_indivCount + 1) >> 1;
		else
			left = island.m_indivCount;

		do
		{
			u64_t* sel = island.m_selected.get();
			shuffle(sel, island.m_selIndivCount, island.m_random);

			u64_t count = std::min(left, island.m_selIndivCount >> 1);
			u64_t step = count & 0b111;

			if (step == 0)
				step = 8;

			while (count)
			{
				m_crossover.init_wave(island);
				m_mutation.init_wave(island);

				__m512d randX = island.m_random.next_512d();
				u32_t cx = _cvtmask8_u32(_mm512_cmplt_pd_mask(randX, prob));

				for (u64_t i = 0; i < step; i++, cx >>= 1)
				{
					u64_t offsetA = *(sel++) * scale;
					u64_t offsetB = *(sel++) * scale;

					dest = xm_step(island, src + offsetA, src + offsetB, dest, static_cast<bool>(cx & 1));
				}

				count -= step;
				left -= step;
				step = 8;
			}

		} while (left);
	}

	template <cc::static_settings S, cc::blueprint<S> B>
	inline f64_t* Algorithm<S, B>::xm_step(Island<S>& island, f64_t* a, f64_t* b, f64_t* out, bool crossover) requires (s_fusedXM)
	{
		if (crossover)
			m_crossover.template perform<mutation_t>(island, a, b, out);
		else
		{
			const f64_t* const minPtr = island.m_minDomain.get();
			const f64_t* const maxPtr = island.m_maxDomain.get();
			const u64_t length = island.m_realGenomeLength;
			const f64_t prob = island.m_mutationProb;

			for (u64_t i = 0; i < length; i += g_vectorGenes)
			{
				__m512d min = _mm512_load_pd(minPtr + i);
				__m512d max = _mm512_load_pd(maxPtr + i);

				__m512d genesA = _mm512_load_pd(a + i);
				_mm512_store_pd(out + i, mutation_t::perform(genesA, min, max, prob, island.m_random));

				if constexpr (s_twins)
				{
					__m512d genesB = _mm512_load_pd(b + i);
					_mm512_store_pd(out + i + length, mutation_t::perform(genesB, min, max, prob, island.m_random));
				}
			}
		}

		if constexpr (s_twins)
			return out + (island.m_realGenomeLength << 1);
		else
			return out + island.m_realGenomeLength;
	}

	template <cc::static_settings S, cc::blueprint<S> B>
	inline f64_t* Algorithm<S, B>::xm_step(Island<S>& island, f64_t* a, f64_t* b, f64_t* out, bool crossover) requires (!s_fusedXM)
	{
		if (crossover)
		{
			m_crossover.template perform<void>(island, a, b, out);
			m_mutation.perform(island, out);

			if constexpr (s_twins)
				m_mutation.perform(island, out += island.m_realGenomeLength);
		}
		else
		{
			std::memcpy(out, a, island.m_realGenomeLength * sizeof(f64_t));
			m_mutation.perform(island, out);

			if constexpr (s_twins)
			{
				out += island.m_realGenomeLength;
				std::memcpy(out, b, island.m_realGenomeLength * sizeof(f64_t));
				m_mutation.perform(island, out);
			}
		}

		return out + island.m_realGenomeLength;
	}
}
