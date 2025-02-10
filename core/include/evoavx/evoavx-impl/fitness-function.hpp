#pragma once
#include "global.hpp"

namespace evo
{
	class FitnessFunction
	{
	public:
		virtual ~FitnessFunction() = default;
		virtual u64_t length() const = 0;
		virtual f64_t evaluate(const f64_t* genes, u64_t threadId) = 0;
	};
}
