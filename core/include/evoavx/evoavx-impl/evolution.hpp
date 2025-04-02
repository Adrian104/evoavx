#pragma once
#include "global.hpp"
#include "algorithm.hpp"
#include "component.hpp"
#include "evaluator.hpp"
#include "island-dec.hpp"
#include "statistics.hpp"
#include "random/random.hpp"
#include "random/splitmix64.hpp"
#include "random/xoshiro256pp.hpp"

namespace evo
{
	using DefaultStaticSettings = StaticSettings<Xoshiro256pp<SplitMix64>, RangeAlg::LEMIRE_64_UNBIASED>;

	template <cc::static_settings S = DefaultStaticSettings>
	class Evolution
	{
	private:
		Shared<S> m_shared;
		std::vector<std::unique_ptr<Island<S>>> m_islands;

		Island<S>& get_island(u64_t island);
		const Island<S>& get_island(u64_t island) const;

	public:
		Evolution(u64_t islandCount = 0);
		~Evolution() = default;

		Evolution(const Evolution<S>&) = delete;
		Evolution<S>& operator=(const Evolution<S>&) = delete;

		Evolution(Evolution<S>&&) = delete;
		Evolution<S>& operator=(Evolution<S>&&) = delete;

		std::vector<std::pair<f64_t, f64_t>> get_genome() const noexcept;
		std::pair<u64_t, u64_t> island_get_population(u64_t island) const;
		f64_t island_get_crossover_probability(u64_t island) const;
		f64_t island_get_mutation_probability(u64_t island) const;
		Extremum island_get_extremum(u64_t island) const;
		u64_t get_island_count() const noexcept;
		u64_t get_seed() const noexcept;

		void clear_genome() noexcept;
		void add_gene(f64_t a, f64_t b);
		void set_seed(u64_t seed) noexcept;
		void set_extremum(Extremum extremum);
		void set_population(u64_t total, u64_t selected);
		void set_crossover_probability(f64_t value);
		void set_mutation_probability(f64_t value);
		void island_set_extremum(u64_t island, Extremum extremum);
		void island_set_population(u64_t island, u64_t total, u64_t selected);
		void island_set_crossover_probability(u64_t island, f64_t value);
		void island_set_mutation_probability(u64_t island, f64_t value);

		template <cc::fitness_function F>
		void set_fitness_function();

		template <cc::fitness_function F, typename... Args>
		F& island_set_fitness_function(u64_t island, Args&&... args);

		template <cc::fitness_function F>
		F* island_get_fitness_function(u64_t island);

		template <cc::fitness_function F>
		bool island_is_fitness_function_set(u64_t island) const;

		template <cc::blueprint<S> B>
		void set_blueprint();

		template <cc::blueprint<S> B>
		void island_set_blueprint(u64_t island);

		template <cc::blueprint<S> B>
		auto island_get_selection(u64_t island) -> typename B::template selection_t<S>*;

		template <cc::blueprint<S> B>
		auto island_get_crossover(u64_t island) -> typename B::template crossover_t<S>*;

		template <cc::blueprint<S> B>
		auto island_get_mutation(u64_t island) -> typename B::template mutation_t<S>*;

		template <cc::blueprint<S> B>
		bool island_is_blueprint_set(u64_t island) const;
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline Island<S>& Evolution<S>::get_island(u64_t island)
	{
		if (island >= m_islands.size())
			throw std::invalid_argument("Invalid island identifier");

		return *(m_islands[island]);
	}

	template <cc::static_settings S>
	inline const Island<S>& Evolution<S>::get_island(u64_t island) const
	{
		if (island >= m_islands.size())
			throw std::invalid_argument("Invalid island identifier");

		return *(m_islands[island]);
	}

	template <cc::static_settings S>
	inline Evolution<S>::Evolution(u64_t islandCount)
	{
		if (islandCount == 0)
			islandCount = std::thread::hardware_concurrency();

		m_islands.reserve(islandCount);
		for (u64_t i = 0; i < islandCount; i++)
			m_islands.push_back(std::make_unique<Island<S>>(m_shared, i));
	}

	template <cc::static_settings S>
	inline std::vector<std::pair<f64_t, f64_t>> Evolution<S>::get_genome() const noexcept
	{
		return m_shared.m_genome;
	}

	template <cc::static_settings S>
	inline std::pair<u64_t, u64_t> Evolution<S>::island_get_population(u64_t island) const
	{
		const Island<S>& ref = get_island(island);
		return std::make_pair(ref.m_indivCount, ref.m_selIndivCount);
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::island_get_crossover_probability(u64_t island) const
	{
		return get_island(island).m_crossoverProb;
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::island_get_mutation_probability(u64_t island) const
	{
		return get_island(island).m_mutationProb;
	}

	template <cc::static_settings S>
	inline Extremum Evolution<S>::island_get_extremum(u64_t island) const
	{
		return get_island(island).m_extremum;
	}

	template <cc::static_settings S>
	inline u64_t Evolution<S>::get_island_count() const noexcept
	{
		return m_islands.size();
	}

	template <cc::static_settings S>
	inline u64_t Evolution<S>::get_seed() const noexcept
	{
		return m_shared.m_seed;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::clear_genome() noexcept
	{
		m_shared.m_genome.clear();
	}

	template <cc::static_settings S>
	inline void Evolution<S>::add_gene(f64_t a, f64_t b)
	{
		if (a > b)
			std::swap(a, b);

		m_shared.m_genome.emplace_back(a, b);
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_seed(u64_t seed) noexcept
	{
		m_shared.m_seed = seed;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_extremum(Extremum extremum)
	{
		for (auto& island : m_islands)
			island->m_extremum = extremum;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_population(u64_t total, u64_t selected)
	{
		if (selected >= total)
			throw std::invalid_argument("The number of selected individuals must be smaller than the total number of individuals");

		for (auto& island : m_islands)
		{
			island->m_indivCount = total;
			island->m_selIndivCount = selected;
		}
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_crossover_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		for (auto& island : m_islands)
			island->m_crossoverProb = value;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_mutation_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		for (auto& island : m_islands)
			island->m_mutationProb = value;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::island_set_extremum(u64_t island, Extremum extremum)
	{
		get_island(island).m_extremum = extremum;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::island_set_population(u64_t island, u64_t total, u64_t selected)
	{
		if (selected >= total)
			throw std::invalid_argument("The number of selected individuals must be smaller than the total number of individuals");

		Island<S>& ref = get_island(island);
		ref.m_indivCount = total;
		ref.m_selIndivCount = selected;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::island_set_crossover_probability(u64_t island, f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		get_island(island).m_crossoverProb = value;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::island_set_mutation_probability(u64_t island, f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		get_island(island).m_mutationProb = value;
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline void Evolution<S>::set_fitness_function()
	{
		for (auto& island : m_islands)
			island->m_evaluator.template add_and_use<Evaluator<S, F>>();
	}

	template <cc::static_settings S> template <cc::fitness_function F, typename... Args>
	inline F& Evolution<S>::island_set_fitness_function(u64_t island, Args&&... args)
	{
		return static_cast<F&>(get_island(island).m_evaluator.template add_and_use<Evaluator<S, F>>(std::forward<Args>(args)...));
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline F* Evolution<S>::island_get_fitness_function(u64_t island)
	{
		return static_cast<F*>(get_island(island).m_evaluator.template get<Evaluator<S, F>>());
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline bool Evolution<S>::island_is_fitness_function_set(u64_t island) const
	{
		return get_island(island).m_evaluator.template is_being_used<Evaluator<S, F>>();
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline void Evolution<S>::set_blueprint()
	{
		for (auto& island : m_islands)
			island->m_algorithm.template add_and_use<Algorithm<S, B>>();
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline void Evolution<S>::island_set_blueprint(u64_t island)
	{
		get_island(island).m_algorithm.template add_and_use<Algorithm<S, B>>();
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline auto Evolution<S>::island_get_selection(u64_t island) -> typename B::template selection_t<S>*
	{
		Algorithm<S, B>* const ptr = get_island(island).m_algorithm.template get<Algorithm<S, B>>();
		return ptr != nullptr ? &ptr->m_selection : nullptr;
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline auto Evolution<S>::island_get_crossover(u64_t island) -> typename B::template crossover_t<S>*
	{
		Algorithm<S, B>* const ptr = get_island(island).m_algorithm.template get<Algorithm<S, B>>();
		return ptr != nullptr ? &ptr->m_crossover : nullptr;
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline auto Evolution<S>::island_get_mutation(u64_t island) -> typename B::template mutation_t<S>*
	{
		Algorithm<S, B>* const ptr = get_island(island).m_algorithm.template get<Algorithm<S, B>>();
		return ptr != nullptr ? &ptr->m_mutation : nullptr;
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline bool Evolution<S>::island_is_blueprint_set(u64_t island) const
	{
		return get_island(island).m_algorithm.template is_being_used<Algorithm<S, B>>();
	}
}
