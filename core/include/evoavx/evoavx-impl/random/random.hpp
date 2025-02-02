#pragma once
#include "../global.hpp"

namespace evo::cc
{
	template <typename EngineT>
	concept basic_prng_engine = std::is_nothrow_constructible_v<EngineT, u64_t>
		&& std::semiregular<EngineT> && requires(EngineT engine, u64_t seed)
	{
		{ engine.init(seed) } noexcept;
		{ engine.step() } noexcept;
		{ engine.next() } noexcept -> std::same_as<u64_t>;
	};

	template <typename EngineT>
	concept wide_prng_engine = std::is_nothrow_constructible_v<EngineT, u64_t>
		&& std::semiregular<EngineT> && requires(EngineT engine, u64_t seed)
	{
		{ engine.init(seed) } noexcept;
		{ engine.step() } noexcept;
		{ engine.jump() } noexcept;
		{ engine.next() } noexcept -> std::same_as<__m512i>;
	};
}
