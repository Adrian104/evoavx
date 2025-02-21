#pragma once
#include "global.hpp"
#include "algorithm.hpp"
#include "component.hpp"
#include "evaluator.hpp"
#include "state.hpp"
#include "statistics.hpp"
#include "triplet.hpp"

namespace evo
{
	template <cc::static_settings S = DefaultStaticSettings>
	class Evolution
	{
	private:
		State<S> m_state;
		Statistics<S> m_statistics;
		Component<AlgorithmBase<S>> m_algorithm;

		void verify();
		void init();

	public:
		void run();

		Evolution<S>& clear_genome();
		Evolution<S>& add_gene(f64_t minimum, f64_t maximum);
		Evolution<S>& set_extremum(Extremum extremum);
		Evolution<S>& set_population(u64_t total, u64_t selected);
		Evolution<S>& set_crossover_probability(f64_t value);
		Evolution<S>& set_mutation_probability(f64_t value);

		std::vector<std::pair<f64_t, f64_t>> get_genome() const noexcept;
		std::pair<u64_t, u64_t> get_population() const noexcept;
		f64_t get_crossover_probability() const noexcept;
		f64_t get_mutation_probability() const noexcept;
		Extremum get_extremum() const noexcept;

		template <cc::fitness_function F, typename... Args>
		F& set_fitness_function(Args&&... args);

		template <cc::fitness_function F>
		void remove_fitness_function();

		template <cc::fitness_function F>
		F* get_fitness_function();

		template <cc::triplet<S> T>
		void set_triplet();

		template <cc::triplet<S> T>
		void remove_triplet();

		template <cc::triplet<S> T>
		auto get_selection() -> typename T::template selection_t<S>*;

		template <cc::triplet<S> T>
		auto get_crossover() -> typename T::template crossover_t<S>*;

		template <cc::triplet<S> T>
		auto get_mutation() -> typename T::template mutation_t<S>*;
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline void Evolution<S>::verify()
	{
		m_state.verify();
		if (m_algorithm.get_used() == nullptr)
			throw std::runtime_error("Triplet is not set");
	}

	template <cc::static_settings S>
	inline void Evolution<S>::init()
	{
		m_statistics.start();
		m_state.init();
	}

	template <cc::static_settings S>
	inline void Evolution<S>::run()
	{
		verify();
		init();

		f64_t meanEstimate = m_state.m_evaluator.get_used()->evaluate_population(m_state);
		m_statistics.update(m_state, meanEstimate);
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::clear_genome()
	{
		m_state.m_genome.clear();
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::add_gene(f64_t minimum, f64_t maximum)
	{
		if (minimum >= maximum)
			throw std::invalid_argument("The minimum value must be smaller than the maximum value");

		m_state.m_genome.emplace_back(minimum, maximum);
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_extremum(Extremum extremum)
	{
		m_state.m_extremum = extremum;
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_population(u64_t total, u64_t selected)
	{
		if (selected >= total)
			throw std::invalid_argument("The number of selected individuals must be smaller than the total number of individuals");

		m_state.m_indivCount = total;
		m_state.m_selIndivCount = selected;
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_crossover_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		m_state.m_crossoverProb = value;
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_mutation_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		m_state.m_mutationProb = value;
		return *this;
	}

	template <cc::static_settings S>
	inline std::vector<std::pair<f64_t, f64_t>> Evolution<S>::get_genome() const noexcept
	{
		return m_state.m_genome;
	}

	template <cc::static_settings S>
	inline std::pair<u64_t, u64_t> Evolution<S>::get_population() const noexcept
	{
		return { m_state.m_indivCount, m_state.m_selIndivCount };
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::get_crossover_probability() const noexcept
	{
		return m_state.m_crossoverProb;
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::get_mutation_probability() const noexcept
	{
		return m_state.m_mutationProb;
	}

	template <cc::static_settings S>
	inline Extremum Evolution<S>::get_extremum() const noexcept
	{
		return m_state.m_extremum;
	}

	template <cc::static_settings S> template <cc::fitness_function F, typename... Args>
	inline F& Evolution<S>::set_fitness_function(Args&&... args)
	{
		return static_cast<F&>(m_state.m_evaluator.template add_and_use<Evaluator<S, F>>(std::forward<Args>(args)...));
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline void Evolution<S>::remove_fitness_function()
	{
		m_state.m_evaluator.template remove<Evaluator<S, F>>();
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline F* Evolution<S>::get_fitness_function()
	{
		return static_cast<F*>(m_state.m_evaluator.template get<Evaluator<S, F>>());
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline void Evolution<S>::set_triplet()
	{
		m_algorithm.template add_and_use<Algorithm<S, T>>();
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline void Evolution<S>::remove_triplet()
	{
		m_algorithm.template remove<Algorithm<S, T>>();
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline auto Evolution<S>::get_selection() -> typename T::template selection_t<S>*
	{
		Algorithm<S, T>* const ptr = m_algorithm.template get<Algorithm<S, T>>();
		return ptr != nullptr ? &ptr->m_selection : nullptr;
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline auto Evolution<S>::get_crossover() -> typename T::template crossover_t<S>*
	{
		Algorithm<S, T>* const ptr = m_algorithm.template get<Algorithm<S, T>>();
		return ptr != nullptr ? &ptr->m_crossover : nullptr;
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline auto Evolution<S>::get_mutation() -> typename T::template mutation_t<S>*
	{
		Algorithm<S, T>* const ptr = m_algorithm.template get<Algorithm<S, T>>();
		return ptr != nullptr ? &ptr->m_mutation : nullptr;
	}
}
