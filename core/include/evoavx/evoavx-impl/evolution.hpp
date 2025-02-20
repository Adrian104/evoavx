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
	class Evolution : private State<S>
	{
	private:
		using state = State<S>;

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
		if (m_algorithm.get_used() == nullptr)
			throw std::runtime_error("Triplet is not set");

		if (state::m_evaluator.get_used() == nullptr)
			throw std::runtime_error("Fitness function is not set");

		if (state::m_genome.empty())
			throw std::runtime_error("Genome is empty");

		if (state::m_indivCount == 0 || state::m_selIndivCount == 0)
			throw std::runtime_error("Population size is not set");

		if (state::m_evaluator.get_used()->get_length() != state::m_genome.size())
			throw std::runtime_error("Genome length does not match the number of arguments required by the fitness function");
	}

	template <cc::static_settings S>
	inline void Evolution<S>::init()
	{
		state::m_geneCount = state::m_indivCount * state::m_genome.size();
		state::m_selGeneCount = state::m_selIndivCount * state::m_genome.size();

		u64_t genesTotal = alignment_ceil<f64_t>(state::m_geneCount);
		u64_t selectedTotal = alignment_ceil<f64_t>(state::m_selGeneCount);
		u64_t scoresTotal = alignment_ceil<f64_t>(state::m_indivCount);

		state::m_genes = unique<f64_t[]>(allocate<f64_t>(genesTotal));
		state::m_selected = unique<f64_t[]>(allocate<f64_t>(selectedTotal));
		state::m_scores = unique<f64_t[]>(allocate<f64_t>(scoresTotal));

		u64_t domainArraySize = state::m_genome.size() + g_vectorGenes - 1;
		u64_t domainTotal = alignment_ceil<f64_t>(domainArraySize);

		state::m_minDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));
		state::m_maxDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));
		state::m_diffDomain = unique<f64_t[]>(allocate<f64_t>(domainTotal));

		std::random_device rd;
		state::m_random.init(rd());

		for (u64_t i = 0, j = 0; i < domainArraySize; i++, j++)
		{
			if (j >= state::m_genome.size())
				j -= state::m_genome.size();

			auto [a, b] = state::m_genome[j];

			state::m_minDomain[i] = a;
			state::m_maxDomain[i] = b;
			state::m_diffDomain[i] = b - a;
		}

		u64_t jcrr = 0;
		u64_t jnext = 0;

		for (u64_t i = 0; i < genesTotal; i += g_vectorGenes)
		{
			jcrr = jnext;
			jnext = (i + g_vectorGenes) % state::m_genome.size();

			__m512d random = state::m_random.next_512d();
			__m512d min = _mm512_loadu_pd(state::m_minDomain.get() + jcrr);
			__m512d diff = _mm512_loadu_pd(state::m_diffDomain.get() + jcrr);
			__m512d initValues = _mm512_fmadd_pd(diff, random, min);

			_mm512_store_pd(state::m_genes.get() + i, initValues);
		}
	}

	template <cc::static_settings S>
	inline void Evolution<S>::run()
	{
		m_statistics.start();

		verify();
		init();

		f64_t meanEstimate = state::m_evaluator.get_used()->evaluate_population(*this);
		m_statistics.update(*this, meanEstimate);
	}

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

		state::m_indivCount = total;
		state::m_selIndivCount = selected;
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
		return { state::m_indivCount, state::m_selIndivCount };
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

	template <cc::static_settings S> template <cc::fitness_function F, typename... Args>
	inline F& Evolution<S>::set_fitness_function(Args&&... args)
	{
		return static_cast<F&>(state::m_evaluator.add_and_use<Evaluator<S, F>>(std::forward<Args>(args)...));
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline void Evolution<S>::remove_fitness_function()
	{
		state::m_evaluator.remove<Evaluator<S, F>>();
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline F* Evolution<S>::get_fitness_function()
	{
		return static_cast<F*>(state::m_evaluator.get<Evaluator<S, F>>());
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
