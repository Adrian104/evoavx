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
		{ engine.next_512i() } noexcept -> std::same_as<__m512i>;
	};
}

namespace evo
{
	template <cc::wide_prng_engine EngineT>
	class Random : public EngineT
	{
	public:
		__m512d next_512d() noexcept;
	};
}

namespace evo
{
	template <cc::wide_prng_engine EngineT>
	inline __m512d Random<EngineT>::next_512d() noexcept
	{
		__m512i source = EngineT::next_512i();
		__m512i shifted = _mm512_srli_epi64(source, 11);
		__m512d converted = _mm512_cvtepu64_pd(shifted);

		return _mm512_mul_pd(converted, _mm512_set1_pd(0x1.0p-53));
	}
}
