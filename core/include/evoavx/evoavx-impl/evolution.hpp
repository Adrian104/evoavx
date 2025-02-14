#pragma once
#include "global.hpp"
#include "algorithm.hpp"
#include "component.hpp"
#include "state.hpp"
#include "triplet.hpp"

namespace evo
{
	template <cc::static_settings S = DefaultStaticSettings>
	class Evolution : private State<S>
	{
	private:
		using state = State<S>;
		Component<AlgorithmBase<S>> m_algorithm;

	public:
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

		template <cc::inherits_from<FitnessFunction> T, typename... Args>
		T& set_fitness_function(Args&&... args);

		template <cc::inherits_from<FitnessFunction> T>
		void remove_fitness_function();

		template <cc::inherits_from<FitnessFunction> T>
		T* get_fitness_function();

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
	inline Evolution<S>& Evolution<S>::clear_genome()
	{
		state::m_genome.clear();
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::add_gene(f64_t minimum, f64_t maximum)
	{
		if (minimum >= maximum)
			throw std::invalid_argument("The minimum value must be smaller than the maximum value");

		state::m_genome.emplace_back(minimum, maximum);
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_extremum(Extremum extremum)
	{
		state::m_extremum = extremum;
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_population(u64_t total, u64_t selected)
	{
		if (selected >= total)
			throw std::invalid_argument("The number of selected individuals must be smaller than the total number of individuals");

		state::m_individualCount = total;
		state::m_selectedIndividualCount = selected;
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_crossover_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		state::m_crossoverProb = value;
		return *this;
	}

	template <cc::static_settings S>
	inline Evolution<S>& Evolution<S>::set_mutation_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		state::m_mutationProb = value;
		return *this;
	}

	template <cc::static_settings S>
	inline std::vector<std::pair<f64_t, f64_t>> Evolution<S>::get_genome() const noexcept
	{
		return state::m_genome;
	}

	template <cc::static_settings S>
	inline std::pair<u64_t, u64_t> Evolution<S>::get_population() const noexcept
	{
		return { state::m_individualCount, state::m_selectedIndividualCount };
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::get_crossover_probability() const noexcept
	{
		return state::m_crossoverProb;
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::get_mutation_probability() const noexcept
	{
		return state::m_mutationProb;
	}

	template <cc::static_settings S>
	inline Extremum Evolution<S>::get_extremum() const noexcept
	{
		return state::m_extremum;
	}

	template <cc::static_settings S> template <cc::inherits_from<FitnessFunction> T, typename... Args>
	inline T& Evolution<S>::set_fitness_function(Args&&... args)
	{
		return state::m_fitnessFunc.add_and_use<T>(std::forward<Args>(args)...);
	}

	template <cc::static_settings S> template <cc::inherits_from<FitnessFunction> T>
	inline void Evolution<S>::remove_fitness_function()
	{
		state::m_fitnessFunc.remove<T>();
	}

	template <cc::static_settings S> template <cc::inherits_from<FitnessFunction> T>
	inline T* Evolution<S>::get_fitness_function()
	{
		return state::m_fitnessFunc.get<T>();
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline void Evolution<S>::set_triplet()
	{
		m_algorithm.add_and_use<Algorithm<S, T>>();
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline void Evolution<S>::remove_triplet()
	{
		m_algorithm.remove<Algorithm<S, T>>();
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline auto Evolution<S>::get_selection() -> typename T::template selection_t<S>*
	{
		Algorithm<S, T>* const ptr = m_algorithm.get<Algorithm<S, T>>();
		return ptr != nullptr ? &ptr->m_selection : nullptr;
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline auto Evolution<S>::get_crossover() -> typename T::template crossover_t<S>*
	{
		Algorithm<S, T>* const ptr = m_algorithm.get<Algorithm<S, T>>();
		return ptr != nullptr ? &ptr->m_crossover : nullptr;
	}

	template <cc::static_settings S> template <cc::triplet<S> T>
	inline auto Evolution<S>::get_mutation() -> typename T::template mutation_t<S>*
	{
		Algorithm<S, T>* const ptr = m_algorithm.get<Algorithm<S, T>>();
		return ptr != nullptr ? &ptr->m_mutation : nullptr;
	}
}
