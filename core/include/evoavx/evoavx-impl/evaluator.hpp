#pragma once
#include "global.hpp"
#include "static-settings.hpp"

namespace evo::cc
{
	template <typename T>
	concept fitness_function = requires(const T cinstance, T instance, const f64_t* genes)
	{
		{ cinstance.length() } -> std::same_as<u64_t>;
		{ instance.evaluate(genes) } -> std::same_as<f64_t>;
	};
}

namespace evo
{
	template <cc::static_settings S>
	class EvaluatorBase
	{
	public:
		virtual ~EvaluatorBase() = default;
		virtual u64_t get_length() const = 0;
		virtual f64_t evaluate_individual(const f64_t* genes) = 0;
		//virtual f64_t evaluate_population(State<S>& state) = 0;
	};

	template <cc::static_settings S, cc::fitness_function F>
	class Evaluator : public EvaluatorBase<S>, public F
	{
	public:
		template <typename... Args>
		Evaluator(Args&&... args) : F(std::forward<Args>(args)...) {}

		u64_t get_length() const override;
		f64_t evaluate_individual(const f64_t* genes) override;
		//f64_t evaluate_population(State<S>& state) override;
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

	/*template <cc::static_settings S, cc::fitness_function F>
	inline f64_t Evaluator<S, F>::evaluate_population(State<S>& state)
	{
		u64_t genomeSize = state.m_genome.size();
		f64_t totalScore = 0;

		for (u64_t i = 0; i < state.m_indivCount; i++)
		{
			f64_t score = F::evaluate(state.m_genes.get() + i * genomeSize);
			state.m_scores[i] = score;
			totalScore += score;
		}

		return totalScore / state.m_indivCount;
	}*/
}
