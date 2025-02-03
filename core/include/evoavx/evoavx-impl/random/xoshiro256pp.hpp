#pragma once
#include "random.hpp"

namespace evo
{
	template <cc::basic_prng_engine SeederT>
	class Xoshiro256pp
	{
	private:
		__m512i m_state[4]{};
		void jump(const std::array<u64_t, 4>& coefficients) noexcept;

	public:
		Xoshiro256pp() noexcept = default;
		Xoshiro256pp(u64_t seed) noexcept;

		void init(u64_t seed) noexcept;
		void step() noexcept;
		void jump() noexcept;
		__m512i next_512i() noexcept;
	};
}

namespace evo
{
	template <cc::basic_prng_engine SeederT>
	inline void Xoshiro256pp<SeederT>::jump(const std::array<u64_t, 4>& coefficients) noexcept
	{
		__m512i s0 = _mm512_setzero_si512();
		__m512i s1 = _mm512_setzero_si512();
		__m512i s2 = _mm512_setzero_si512();
		__m512i s3 = _mm512_setzero_si512();

		for (int i = 0; i < 4; i++)
		{
			const u64_t c = coefficients[i];
			for (int b = 0; b < 64; b++)
			{
				__mmask8 mask = _cvtu32_mask8(-(i64_t)((c >> b) & 1ULL));

				s0 = _mm512_mask_xor_epi64(s0, mask, s0, m_state[0]);
				s1 = _mm512_mask_xor_epi64(s1, mask, s1, m_state[1]);
				s2 = _mm512_mask_xor_epi64(s2, mask, s2, m_state[2]);
				s3 = _mm512_mask_xor_epi64(s3, mask, s3, m_state[3]);

				step();
			}
		}

		m_state[0] = s0;
		m_state[1] = s1;
		m_state[2] = s2;
		m_state[3] = s3;
	}

	template <cc::basic_prng_engine SeederT>
	inline Xoshiro256pp<SeederT>::Xoshiro256pp(u64_t seed) noexcept
	{
		init(seed);
	}

	template <cc::basic_prng_engine SeederT>
	inline void Xoshiro256pp<SeederT>::init(u64_t seed) noexcept
	{
		SeederT seeder(seed);

		m_state[0] = _mm512_set1_epi64(seeder.next());
		m_state[1] = _mm512_set1_epi64(seeder.next());
		m_state[2] = _mm512_set1_epi64(seeder.next());
		m_state[3] = _mm512_set1_epi64(seeder.next());

		__mmask8 mask = _cvtu32_mask8(0xFF);

		__m512i s0 = _mm512_setzero_si512();
		__m512i s1 = _mm512_setzero_si512();
		__m512i s2 = _mm512_setzero_si512();
		__m512i s3 = _mm512_setzero_si512();

		for (int r = 0; r < 8; r++)
		{
			s0 = _mm512_mask_mov_epi64(s0, mask, m_state[0]);
			s1 = _mm512_mask_mov_epi64(s1, mask, m_state[1]);
			s2 = _mm512_mask_mov_epi64(s2, mask, m_state[2]);
			s3 = _mm512_mask_mov_epi64(s3, mask, m_state[3]);

			mask = _kshiftli_mask8(mask, 1);
			jump({ 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 });
		}

		m_state[0] = s0;
		m_state[1] = s1;
		m_state[2] = s2;
		m_state[3] = s3;
	}

	template <cc::basic_prng_engine SeederT>
	inline void Xoshiro256pp<SeederT>::step() noexcept
	{
		__m512i t = _mm512_slli_epi64(m_state[1], 17);
		m_state[2] = _mm512_xor_epi64(m_state[2], m_state[0]);
		m_state[3] = _mm512_xor_epi64(m_state[3], m_state[1]);
		m_state[1] = _mm512_xor_epi64(m_state[1], m_state[2]);
		m_state[0] = _mm512_xor_epi64(m_state[0], m_state[3]);
		m_state[2] = _mm512_xor_epi64(m_state[2], t);
		m_state[3] = _mm512_rol_epi64(m_state[3], 45);
	}

	template <cc::basic_prng_engine SeederT>
	inline void Xoshiro256pp<SeederT>::jump() noexcept
	{
		jump({ 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c });
	}

	template <cc::basic_prng_engine SeederT>
	inline __m512i Xoshiro256pp<SeederT>::next_512i() noexcept
	{
		__m512i result = _mm512_add_epi64(m_state[0], m_state[3]);
		result = _mm512_rol_epi64(result, 23);
		result = _mm512_add_epi64(result, m_state[0]);

		step();
		return result;
	}
}
