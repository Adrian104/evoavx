#include "pch.hpp"
#include <evoavx/evoavx.hpp>

TEST_CASE("Function shuffle() works as expected")
{
	evo::Random<evo::DefaultStaticSettings> random;
	random.init(42);

	for (evo::u64_t size = 1ULL << 5; size < (1ULL << 5) + 8; size++)
	{
		std::vector<evo::u16_t> data(size);
		std::vector<evo::u16_t> used(size);
		std::vector<evo::f64_t> sum(size);
		std::vector<evo::u16_t> min(size, std::numeric_limits<evo::u16_t>::max());
		std::vector<evo::u16_t> max(size, std::numeric_limits<evo::u16_t>::min());

		for (evo::u64_t i = 0; i < size; i++)
			data[i] = static_cast<evo::u16_t>(i << 2);

		const evo::u64_t sampleSize = 1ULL << 10;
		for (evo::u64_t sample = 0; sample < sampleSize; sample++)
		{
			evo::shuffle(data.data(), size, random);
			for (evo::u64_t i = 0; i < size; i++)
			{
				sum[i] += static_cast<evo::f64_t>(data[i]);
				min[i] = std::min(min[i], data[i]);
				max[i] = std::max(max[i], data[i]);
			}
		}

		for (auto& value : data)
			used[value >> 2]++;

		for (auto& value : used)
			REQUIRE(value == 1);

		for (evo::u64_t i = 0; i < size; i++)
		{
			REQUIRE(min[i] < 16);
			REQUIRE(max[i] >= (size << 2) - 16);
			REQUIRE_THAT(sum[i] / sampleSize, Catch::Matchers::WithinRel((size << 2) / 2.0, 0.2));
		}
	}
}

TEST_CASE("Function derange() works as expected")
{
	evo::Random<evo::DefaultStaticSettings> random;
	random.init(42);

	for (evo::u64_t size = 1ULL << 5; size < (1ULL << 5) + 8; size++)
	{
		std::vector<evo::u16_t> data(size);
		std::vector<evo::u16_t> prev(size);
		std::vector<evo::u16_t> used(size);
		std::vector<evo::f64_t> sum(size);
		std::vector<evo::u16_t> min(size, std::numeric_limits<evo::u16_t>::max());
		std::vector<evo::u16_t> max(size, std::numeric_limits<evo::u16_t>::min());

		for (evo::u64_t i = 0; i < size; i++)
			data[i] = static_cast<evo::u16_t>(i << 2);

		const evo::u64_t sampleSize = 1ULL << 10;
		bool deranged = true;

		for (evo::u64_t sample = 0; sample < sampleSize; sample++)
		{
			prev = data;
			evo::derange(data.data(), size, random);

			for (evo::u64_t i = 0; i < size; i++)
			{
				if (data[i] == prev[i])
					deranged = false;

				sum[i] += static_cast<evo::f64_t>(data[i]);
				min[i] = std::min(min[i], data[i]);
				max[i] = std::max(max[i], data[i]);
			}
		}

		REQUIRE(deranged);
		for (auto& value : data)
			used[value >> 2]++;

		for (auto& value : used)
			REQUIRE(value == 1);

		for (evo::u64_t i = 0; i < size; i++)
		{
			REQUIRE(min[i] < 16);
			REQUIRE(max[i] >= (size << 2) - 16);
			REQUIRE_THAT(sum[i] / sampleSize, Catch::Matchers::WithinRel((size << 2) / 2.0, 0.2));
		}
	}
}
