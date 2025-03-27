#pragma once
#include "global.hpp"
#include "island-dec.hpp"

namespace evo
{
	template <cc::static_settings S>
	class EvaluatorBase
	{
	public:
		virtual ~EvaluatorBase() = default;
		virtual u64_t get_length() const = 0;
		virtual f64_t evaluate_individual(const f64_t* genes) = 0;
		virtual f64_t evaluate_population(Island<S>& island) = 0;
		virtual f64_t evaluate_population_with_aux(Island<S>& island) = 0;
	};

	template <cc::static_settings S, cc::fitness_function F>
	class Evaluator : public EvaluatorBase<S>, public F
	{
	public:
		template <typename... Args>
		Evaluator(Args&&... args) : F(std::forward<Args>(args)...) {}

		u64_t get_length() const override;
		f64_t evaluate_individual(const f64_t* genes) override;
		f64_t evaluate_population(Island<S>& island) override;
		f64_t evaluate_population_with_aux(Island<S>& island) override;
	};
}

namespace evo
{
	template <cc::static_settings S, cc::fitness_function F>
	inline u64_t Evaluator<S, F>::get_length() const
	{
		return F::length();
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_individual(const f64_t* genes)
	{
		return F::evaluate(genes);
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
			const f64_t score = F::evaluate(genes);

			genes += step;
			totalScore += score;
			*(scores++) = score;
		}

		return totalScore / island.m_indivCount;
	}

	template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_population_with_aux(Island<S>& island)
	{
		const u64_t step = island.m_realGenomeLength;
		const f64_t* genes = island.m_current.get();
		const f64_t* const end = genes + island.m_indivCount * step;

		f64_t totalScore = 0;
		f64_t* scores = island.m_scores.get();
		f64_t* scoresAux = island.m_scoresAux.get();

		while (genes != end)
		{
			const f64_t score = F::evaluate(genes);

			genes += step;
			totalScore += score;
			*(scores++) = score;
			*(scoresAux++) = score;
		}

		return totalScore / island.m_indivCount;
	}
}
