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
	using DefaultStaticSettings = StaticSettings<Xoshiro256pp<SplitMix64>, RangeAlg::LEMIRE_64_UNBIASED,
		FusedXM::AUTO, ForceDomain::AUTO, Cache::ENABLED_MURMUR_HASH_64A>;

	template <cc::static_settings S = DefaultStaticSettings>
	class Evolution : private Island<S>
	{
	public:
		Evolution() = default;
		~Evolution() = default;

		Evolution(const Evolution<S>&) = delete;
		Evolution<S>& operator=(const Evolution<S>&) = delete;

		Evolution(Evolution<S>&&) = delete;
		Evolution<S>& operator=(Evolution<S>&&) = delete;

		const std::vector<std::pair<f64_t, f64_t>>& get_genome() const noexcept;
		std::tuple<u64_t, u64_t, u64_t> get_population() const noexcept;
		f64_t get_crossover_probability() const noexcept;
		f64_t get_mutation_probability() const noexcept;
		u64_t get_migration_interval() const noexcept;
		MPI_Comm get_communicator() const noexcept;
		Extremum get_extremum() const noexcept;
		u64_t get_seed() const noexcept;

		void clear_genome() noexcept;
		void add_gene(f64_t a, f64_t b);
		void set_seed(u64_t seed) noexcept;
		void set_extremum(Extremum extremum) noexcept;
		void set_communicator(MPI_Comm communicator) noexcept;
		void set_population(u64_t total, u64_t selected, u64_t migrants);
		void set_migration_interval(u64_t interval);
		void set_crossover_probability(f64_t value);
		void set_mutation_probability(f64_t value);

		u32_t get_cache_size_exponent() const noexcept requires (S::cache_v != Cache::DISABLED);
		void set_cache_size_exponent(u32_t exponent) requires (S::cache_v != Cache::DISABLED);

		template <cc::inspector I, typename... Args>
		I& set_inspector(Args&&... args);

		template <cc::inspector I>
		I* get_inspector() noexcept;

		template <cc::inspector I>
		bool is_inspector_set() const noexcept;

		template <cc::fitness_function F, typename... Args>
		F& set_fitness_function(Args&&... args);

		template <cc::fitness_function F>
		F* get_fitness_function() noexcept;

		template <cc::fitness_function F>
		bool is_fitness_function_set() const noexcept;

		template <cc::blueprint<S> B>
		void set_blueprint();

		template <cc::blueprint<S> B>
		auto get_selection() noexcept -> typename B::template selection_t<S>*;

		template <cc::blueprint<S> B>
		auto get_crossover() noexcept -> typename B::template crossover_t<S>*;

		template <cc::blueprint<S> B>
		auto get_mutation() noexcept -> typename B::template mutation_t<S>*;

		template <cc::blueprint<S> B>
		bool is_blueprint_set() const noexcept;

		Result run();
	};
}

namespace evo
{
	template <cc::static_settings S>
	inline const std::vector<std::pair<f64_t, f64_t>>& Evolution<S>::get_genome() const noexcept
	{
		return Island<S>::m_genome;
	}

	template <cc::static_settings S>
	inline std::tuple<u64_t, u64_t, u64_t> Evolution<S>::get_population() const noexcept
	{
		return std::make_tuple(Island<S>::m_indivCount, Island<S>::m_selIndivCount, Island<S>::m_migIndivCount);
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::get_crossover_probability() const noexcept
	{
		return Island<S>::m_crossoverProb;
	}

	template <cc::static_settings S>
	inline f64_t Evolution<S>::get_mutation_probability() const noexcept
	{
		return Island<S>::m_mutationProb;
	}

	template <cc::static_settings S>
	inline u64_t Evolution<S>::get_migration_interval() const noexcept
	{
		return Island<S>::m_migInterval;
	}

	template <cc::static_settings S>
	inline MPI_Comm Evolution<S>::get_communicator() const noexcept
	{
		return Island<S>::m_communicator;
	}

	template <cc::static_settings S>
	inline Extremum Evolution<S>::get_extremum() const noexcept
	{
		return Island<S>::m_extremum;
	}

	template <cc::static_settings S>
	inline u64_t Evolution<S>::get_seed() const noexcept
	{
		return Island<S>::m_seed;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::clear_genome() noexcept
	{
		Island<S>::m_genome.clear();
	}

	template <cc::static_settings S>
	inline void Evolution<S>::add_gene(f64_t a, f64_t b)
	{
		if (a > b)
			std::swap(a, b);

		Island<S>::m_genome.emplace_back(a, b);
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_seed(u64_t seed) noexcept
	{
		Island<S>::m_seed = seed;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_extremum(Extremum extremum) noexcept
	{
		Island<S>::m_extremum = extremum;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_communicator(MPI_Comm communicator) noexcept
	{
		Island<S>::m_communicator = communicator;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_population(u64_t total, u64_t selected, u64_t migrants)
	{
		if (total == 0 || selected == 0 || migrants == 0)
			throw std::invalid_argument("The number of individuals must be greater than zero");

		if (total <= migrants)
			throw std::invalid_argument("The number of migrants must be smaller than the total number of individuals");

		Island<S>::m_indivCount = total;
		Island<S>::m_selIndivCount = selected;
		Island<S>::m_migIndivCount = migrants;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_migration_interval(u64_t interval)
	{
		if (interval == 0)
			throw std::invalid_argument("Interval must be greater than zero");

		Island<S>::m_migInterval = interval;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_crossover_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		Island<S>::m_crossoverProb = value;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_mutation_probability(f64_t value)
	{
		if (value < 0.0 || value > 1.0)
			throw std::invalid_argument("The probability must be within the range [0, 1]");

		Island<S>::m_mutationProb = value;
	}

	template <cc::static_settings S>
	inline u32_t Evolution<S>::get_cache_size_exponent() const noexcept requires (S::cache_v != Cache::DISABLED)
	{
		return Island<S>::m_cacheExponent;
	}

	template <cc::static_settings S>
	inline void Evolution<S>::set_cache_size_exponent(u32_t exponent) requires (S::cache_v != Cache::DISABLED)
	{
		if (exponent > 63)
			throw std::invalid_argument("The exponent must be within the range [0, 63]");

		Island<S>::m_cacheExponent = exponent;
	}

	template <cc::static_settings S> template <cc::inspector I, typename... Args>
	inline I& Evolution<S>::set_inspector(Args&&... args)
	{
		return Island<S>::m_inspector.template add_and_use<I>(std::forward<Args>(args)...);
	}

	template <cc::static_settings S> template <cc::inspector I>
	inline I* Evolution<S>::get_inspector() noexcept
	{
		return Island<S>::m_inspector.template get<I>();
	}

	template <cc::static_settings S> template <cc::inspector I>
	inline bool Evolution<S>::is_inspector_set() const noexcept
	{
		return Island<S>::m_inspector.template is_being_used<I>();
	}

	template <cc::static_settings S> template <cc::fitness_function F, typename... Args>
	inline F& Evolution<S>::set_fitness_function(Args&&... args)
	{
		return static_cast<F&>(Island<S>::m_evaluator.template add_and_use<Evaluator<S, F>>(std::forward<Args>(args)...));
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline F* Evolution<S>::get_fitness_function() noexcept
	{
		return static_cast<F*>(Island<S>::m_evaluator.template get<Evaluator<S, F>>());
	}

	template <cc::static_settings S> template <cc::fitness_function F>
	inline bool Evolution<S>::is_fitness_function_set() const noexcept
	{
		return Island<S>::m_evaluator.template is_being_used<Evaluator<S, F>>();
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline void Evolution<S>::set_blueprint()
	{
		Island<S>::m_algorithm.template add_and_use<Algorithm<S, B>>();
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline auto Evolution<S>::get_selection() noexcept -> typename B::template selection_t<S>*
	{
		Algorithm<S, B>* const ptr = Island<S>::m_algorithm.template get<Algorithm<S, B>>();
		return ptr != nullptr ? &ptr->m_selection : nullptr;
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline auto Evolution<S>::get_crossover() noexcept -> typename B::template crossover_t<S>*
	{
		Algorithm<S, B>* const ptr = Island<S>::m_algorithm.template get<Algorithm<S, B>>();
		return ptr != nullptr ? &ptr->m_crossover : nullptr;
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline auto Evolution<S>::get_mutation() noexcept -> typename B::template mutation_t<S>*
	{
		Algorithm<S, B>* const ptr = Island<S>::m_algorithm.template get<Algorithm<S, B>>();
		return ptr != nullptr ? &ptr->m_mutation : nullptr;
	}

	template <cc::static_settings S> template <cc::blueprint<S> B>
	inline bool Evolution<S>::is_blueprint_set() const noexcept
	{
		return Island<S>::m_algorithm.template is_being_used<Algorithm<S, B>>();
	}

	template <cc::static_settings S>
	inline Result Evolution<S>::run()
	{
		return Island<S>::entry_point();
	}
}
