#pragma once
#include "global.hpp"
#include "island-dec.hpp"

namespace evo
{
	template <cc::static_settings S>
	class EvaluatorBase
	{
	protected:
		u64_t m_mask = 0;
		u64_t m_length = 0;
		unique<f64_t[]> m_keys;
		unique<f64_t[]> m_values;

		static u64_t hash(const f64_t* genes, u64_t length) noexcept requires (S::cache_v == Cache::ENABLED_CRC_32);
		static u64_t hash(const f64_t* genes, u64_t length) noexcept requires (S::cache_v == Cache::ENABLED_MURMUR_HASH_64A);

	public:
		void init_cache(u32_t power, u64_t genomeLength);

		virtual ~EvaluatorBase() = default;
		virtual u64_t get_length() const = 0;
		virtual f64_t evaluate_individual(const f64_t* genes) = 0;
		virtual f64_t evaluate_population(Island<S>& island) = 0;
		virtual void evaluate_subset(Island<S>& island, u64_t count) = 0;
	};

	template <cc::static_settings S, cc::fitness_function F>
	class Evaluator : public EvaluatorBase<S>, public F
	{
	private:
		f64_t evaluate_impl(const f64_t* genes) requires (S::cache_v != Cache::DISABLED);
		f64_t evaluate_impl(const f64_t* genes) requires (S::cache_v == Cache::DISABLED);

	public:
		template <typename... Args>
		Evaluator(Args&&... args) : F(std::forward<Args>(args)...) {}

		u64_t get_length() const override;
		f64_t evaluate_individual(const f64_t* genes) override;
		f64_t evaluate_population(Island<S>& island) override;
		void evaluate_subset(Island<S>& island, u64_t count) override;
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline u64_t EvaluatorBase<S>::hash(const f64_t* genes, u64_t length) noexcept requires (S::cache_v == Cache::ENABLED_CRC_32)
	{
		u64_t hash = 0xC6A4A7935BD1E995ULL;
		const f64_t* const end = genes + length;

		while (genes != end)
		{
			u64_t crr = *reinterpret_cast<const u64_t*>(genes++);
			hash = _mm_crc32_u64(hash, crr);
		}

		return hash;
	}

	template <cc::static_settings S>
	inline u64_t EvaluatorBase<S>::hash(const f64_t* genes, u64_t length) noexcept requires (S::cache_v == Cache::ENABLED_MURMUR_HASH_64A)
	{
		constexpr static u64_t s_mul = 0xC6A4A7935BD1E995ULL;
		constexpr static u32_t s_shift = 47;

		u64_t hash = length * s_mul;
		const f64_t* const end = genes + length;

		while (genes != end)
		{
			u64_t crr = *reinterpret_cast<const u64_t*>(genes++);

			crr *= s_mul;
			crr ^= crr >> s_shift;
			crr *= s_mul;

			hash ^= crr;
			hash *= s_mul;
		}

		hash ^= hash >> s_shift;
		hash *= s_mul;
		hash ^= hash >> s_shift;

		return hash;
	}

	template <cc::static_settings S>
	inline void EvaluatorBase<S>::init_cache(u32_t power, u64_t genomeLength)
	{
		if constexpr (S::cache_v != Cache::DISABLED)
		{
			const u64_t count = 1ULL << power;
			const u64_t keysCap = alignment_ceil<f64_t>(count * genomeLength);
			const u64_t valuesCap = alignment_ceil<f64_t>(count);

			m_mask = count - 1ULL;
			m_length = genomeLength;
			m_keys = unique<f64_t[]>(allocate<f64_t>(keysCap));
			m_values = unique<f64_t[]>(allocate<f64_t>(valuesCap));

			for (u64_t i = 0; i < keysCap; i++)
				m_keys[i] = std::numeric_limits<f64_t>::quiet_NaN();

			for (u64_t i = 0; i < valuesCap; i++)
				m_values[i] = std::numeric_limits<f64_t>::quiet_NaN();
		}
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_impl(const f64_t* genes) requires (S::cache_v != Cache::DISABLED)
	{
		using scope = EvaluatorBase<S>;

		u64_t length = scope::m_length;
		u64_t bytes = length * sizeof(f64_t);
		u64_t index = scope::hash(genes, length) & scope::m_mask;
		f64_t* key = scope::m_keys.get() + index * length;
		f64_t* value = scope::m_values.get() + index;

		if (std::memcmp(genes, key, bytes) == 0)
			return *value;

		f64_t score = F::evaluate(genes);
		std::memcpy(key, genes, bytes);
		*value = score;

		return score;
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_impl(const f64_t* genes) requires (S::cache_v == Cache::DISABLED)
	{
		return F::evaluate(genes);
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline u64_t Evaluator<S, F>::get_length() const
	{
		return F::length();
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_individual(const f64_t* genes)
	{
		return evaluate_impl(genes);
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_population(Island<S>& island)
	{
		const u64_t step = island.m_realGenomeLength;
		const f64_t* genes = island.m_current.get();
		const f64_t* const end = genes + island.m_indivCount * step;

		f64_t totalScore = 0;
		f64_t* scores = island.m_scores.get();

		while (genes != end)
		{
			const f64_t score = evaluate_impl(genes);

			genes += step;
			totalScore += score;
			*(scores++) = score;
		}

		return totalScore / island.m_indivCount;
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline void Evaluator<S, F>::evaluate_subset(Island<S>& island, u64_t count)
	{
		f64_t* const scores = island.m_scores.get();
		const f64_t* const genes = island.m_current.get();
		const u64_t step = island.m_realGenomeLength;
		const u64_t* ptr = island.m_selected.get();
		const u64_t* const end = ptr + count;

		while (ptr != end)
		{
			const u64_t idx = *(ptr++);
			scores[idx] = evaluate_impl(genes + idx * step);
		}
	}
}
