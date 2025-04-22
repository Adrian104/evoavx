#pragma once
#include "global.hpp"
#include "random/random.hpp"

namespace evo
{
	template <cc::static_settings S, typename T>
	inline void shuffle(T* data, u64_t count, Random<S>& random)
	{
		assert(data != nullptr);
		assert(count > 0);

		u64_t init = (count & ~static_cast<u64_t>(0b111)) + 8;
		alignas(g_vectorBytes) u64_t buffer[8]{ init, init - 1, init - 2, init - 3, init - 4, init - 5, init - 6, init - 7 };

		__m512i ranges = _mm512_load_epi64(buffer);
		__m512i subtract = _mm512_set1_epi64(8);

		_mm512_store_epi64(buffer, random.range_512i(ranges));
		for (u64_t i = 8 - (count & 0b111); i < 8; i++)
			std::swap(data[buffer[i]], data[--count]);

		while (count)
		{
			ranges = _mm512_sub_epi64(ranges, subtract);
			_mm512_store_epi64(buffer, random.range_512i(ranges));

			for (u64_t i = 0; i < 8; i++)
				std::swap(data[buffer[i]], data[--count]);
		}
	}
}
