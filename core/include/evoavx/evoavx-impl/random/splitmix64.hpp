#pragma once
#include "random.hpp"

namespace evo
{
	class SplitMix64
	{
	private:
		u64_t m_state{};

	public:
		SplitMix64() noexcept = default;
		SplitMix64(u64_t seed) noexcept;

		void init(u64_t seed) noexcept;
		void step() noexcept;
		u64_t next() noexcept;
	};
}

namespace evo
{
	inline SplitMix64::SplitMix64(u64_t seed) noexcept
		: m_state(seed) {}

	inline void SplitMix64::init(u64_t seed) noexcept
	{
		m_state = seed;
	}

	inline void SplitMix64::step() noexcept
	{
		m_state += 0x9e3779b97f4a7c15;
	}

	inline u64_t SplitMix64::next() noexcept
	{
		step();
		u64_t z = m_state;

		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}
}
