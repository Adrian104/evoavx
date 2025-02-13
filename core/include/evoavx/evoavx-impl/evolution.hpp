#pragma once
#include "global.hpp"
#include "component.hpp"
#include "operator.hpp"
#include "state.hpp"

namespace evo
{
	template <cc::static_settings S = DefaultStaticSettings>
	class Evolution
	{
		State<S> m_state;
		Component<Selection<S>> m_selection;
		Component<FusedXMBase<S>> m_fusedXM;

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
		T& set(Args&&... args);

		template <template <typename> typename T>
			requires cc::inherits_from<T<S>, Selection<S>>
		T<S>& set();

		template <template <typename> typename X, template <typename> typename M>
			requires cc::crossover<X<S>> && cc::mutation<M<S>>
		FusedXM<S, X, M>& set();

		template <cc::inherits_from<FitnessFunction> T>
		T* get();

		template <template <typename> typename T>
			requires cc::inherits_from<T<S>, Selection<S>>
		T<S>* get();

		template <template <typename> typename X, template <typename> typename M>
			requires cc::crossover<X<S>> && cc::mutation<M<S>>
		FusedXM<S, X, M>* get();

		template <cc::inherits_from<FitnessFunction> T>
		const T* get() const;

		template <template <typename> typename T>
			requires cc::inherits_from<T<S>, Selection<S>>
		const T<S>* get() const;

		template <template <typename> typename X, template <typename> typename M>
			requires cc::crossover<X<S>> && cc::mutation<M<S>>
		const FusedXM<S, X, M>* get() const;
	};
}

namespace evo
{
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

		m_state.m_individualCount = total;
		m_state.m_selectedIndividualCount = selected;
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
		return { m_state.m_individualCount, m_state.m_selectedIndividualCount };
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

	template <cc::static_settings S> template <cc::inherits_from<FitnessFunction> T, typename... Args>
	inline T& Evolution<S>::set(Args&&... args)
	{
		return m_state.m_fitnessFunc.add_and_use<T>(std::forward<Args>(args)...);
	}

	template <cc::static_settings S> template <template <typename> typename T>
		requires cc::inherits_from<T<S>, Selection<S>>
	inline T<S>& Evolution<S>::set()
	{
		return m_selection.add_and_use<T<S>>();
	}

	template <cc::static_settings S> template <template <typename> typename X, template <typename> typename M>
		requires cc::crossover<X<S>> && cc::mutation<M<S>>
	inline FusedXM<S, X, M>& Evolution<S>::set()
	{
		return m_fusedXM.add_and_use<FusedXM<S, X, M>>();
	}

	template <cc::static_settings S> template <cc::inherits_from<FitnessFunction> T>
	inline T* Evolution<S>::get()
	{
		return m_state.m_fitnessFunc.get<T>();
	}

	template <cc::static_settings S> template <template <typename> typename T>
		requires cc::inherits_from<T<S>, Selection<S>>
	inline T<S>* Evolution<S>::get()
	{
		return m_selection.get<T<S>>();
	}

	template <cc::static_settings S> template <template <typename> typename X, template <typename> typename M>
		requires cc::crossover<X<S>> && cc::mutation<M<S>>
	inline FusedXM<S, X, M>* Evolution<S>::get()
	{
		return m_fusedXM.get<FusedXM<S, X, M>>();
	}

	template <cc::static_settings S> template <cc::inherits_from<FitnessFunction> T>
	inline const T* Evolution<S>::get() const
	{
		return m_state.m_fitnessFunc.get<T>();
	}

	template <cc::static_settings S> template <template <typename> typename T>
		requires cc::inherits_from<T<S>, Selection<S>>
	inline const T<S>* Evolution<S>::get() const
	{
		return m_selection.get<T<S>>();
	}

	template <cc::static_settings S> template <template <typename> typename X, template <typename> typename M>
		requires cc::crossover<X<S>> && cc::mutation<M<S>>
	inline const FusedXM<S, X, M>* Evolution<S>::get() const
	{
		return m_fusedXM.get<FusedXM<S, X, M>>();
	}
}
