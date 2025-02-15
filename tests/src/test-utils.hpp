#include "pch.hpp"
#include <evoavx/evoavx.hpp>

TEST_CASE("Wide multiplication works as expected")
{
	__m512i a = _mm512_set_epi64(
		0xBA48BF31BAAD2EFA,
		0x144F1230B586010C,
		0x6D680991E58738A0,
		0x86D0FF41E5CB4F63,
		0x3D99C2AFBA249334,
		0x633D1AC07AF9F6CE,
		0x8AD7D77FA10E03CF,
		0x8F48F5A697E66975
	);

	__m512i b = _mm512_set_epi64(
		0xB86F66EDA473CFAA,
		0xD5F0F7430FFF5323,
		0x0BF64B0F35709679,
		0xFD76DA7F0C9FDF98,
		0x88EBECAA4381D9E0,
		0x96BED86BE4E765E5,
		0x85EA7A27F57412C3,
		0x0D9F11B6AF576806
	);

	__m512i expectedLow = _mm512_set_epi64(
		0x2748144E424B5804,
		0x2731CC32029D08A4,
		0x6A649DB49E1783A0,
		0x4B3DE7687D5B5FC8,
		0x698F5E08A9FBE180,
		0xA2086E29CBDB0C46,
		0xF7B0BE127EBD74AD,
		0x18704F2FFC0100BE
	);

	__m512i expectedHigh = _mm512_set_epi64(
		0x863559DC8DEE1C8C,
		0x10F8E7E1313651D0,
		0x051CBA7671CC54F8,
		0x857B23AAAA2C9611,
		0x20F27485CA0712EE,
		0x3A6FC8EC04C271CE,
		0x48A14E80D209AB33,
		0x079FBCB420818BEE
	);

	__m512i low = _mm512_mullo_epi64(a, b);
	__m512i high = evo::mulhi_512i64(a, b);

	__mmask8 lowEqual = _mm512_cmpeq_epu64_mask(low, expectedLow);
	__mmask8 highEqual = _mm512_cmpeq_epu64_mask(high, expectedHigh);

	REQUIRE(_cvtmask8_u32(lowEqual) == 0xFF);
	REQUIRE(_cvtmask8_u32(highEqual) == 0xFF);
}

TEST_CASE("Functions alignment_floor() and alignment_ceil() work as expected")
{
	SECTION("With u8_t")
	{
		REQUIRE(evo::alignment_floor<evo::u8_t>(0) == 0);
		REQUIRE(evo::alignment_ceil<evo::u8_t>(0) == 0);

		for (evo::u64_t i = 1; i < 64; i++)
		{
			REQUIRE(evo::alignment_floor<evo::u8_t>(i) == 0);
			REQUIRE(evo::alignment_ceil<evo::u8_t>(i) == 64);
		}

		REQUIRE(evo::alignment_floor<evo::u8_t>(64) == 64);
		REQUIRE(evo::alignment_ceil<evo::u8_t>(64) == 64);

		for (evo::u64_t i = 65; i < 128; i++)
		{
			REQUIRE(evo::alignment_floor<evo::u8_t>(i) == 64);
			REQUIRE(evo::alignment_ceil<evo::u8_t>(i) == 128);
		}

		REQUIRE(evo::alignment_floor<evo::u8_t>(128) == 128);
		REQUIRE(evo::alignment_ceil<evo::u8_t>(128) == 128);

		for (evo::u64_t i = 129; i < 192; i++)
		{
			REQUIRE(evo::alignment_floor<evo::u8_t>(i) == 128);
			REQUIRE(evo::alignment_ceil<evo::u8_t>(i) == 192);
		}
	}

	SECTION("With f32_t")
	{
		REQUIRE(evo::alignment_floor<evo::f32_t>(0) == 0);
		REQUIRE(evo::alignment_ceil<evo::f32_t>(0) == 0);

		for (evo::u64_t i = 1; i < 16; i++)
		{
			REQUIRE(evo::alignment_floor<evo::f32_t>(i) == 0);
			REQUIRE(evo::alignment_ceil<evo::f32_t>(i) == 16);
		}

		REQUIRE(evo::alignment_floor<evo::f32_t>(16) == 16);
		REQUIRE(evo::alignment_ceil<evo::f32_t>(16) == 16);

		for (evo::u64_t i = 17; i < 32; i++)
		{
			REQUIRE(evo::alignment_floor<evo::f32_t>(i) == 16);
			REQUIRE(evo::alignment_ceil<evo::f32_t>(i) == 32);
		}

		REQUIRE(evo::alignment_floor<evo::f32_t>(32) == 32);
		REQUIRE(evo::alignment_ceil<evo::f32_t>(32) == 32);

		for (evo::u64_t i = 33; i < 48; i++)
		{
			REQUIRE(evo::alignment_floor<evo::f32_t>(i) == 32);
			REQUIRE(evo::alignment_ceil<evo::f32_t>(i) == 48);
		}
	}
}
