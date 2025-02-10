#pragma once
#include "global.hpp"
#include "component.hpp"
#include "operator.hpp"
#include "state.hpp"

namespace evo
{
	template <cc::static_settings S = DefaultStaticSettings>
	class Evolution
	{
		State<S> m_state;
		Component<Selection<S>> m_selection;
		Component<FusedXMBase<S>> m_fusedXM;
	};
}
