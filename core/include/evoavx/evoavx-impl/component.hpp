#pragma once
#include "global.hpp"

namespace evo
{
	template <typename BaseT>
	class Component
	{
	private:
		inline static u64_t s_next = 0;
		BaseT* m_current = nullptr;
		std::vector<std::pair<std::unique_ptr<BaseT>, void*>> m_data;

		template <cc::inherits_from<BaseT> DerivedT>
		static u64_t get_id() noexcept;

	public:
		Component() = default;
		~Component() = default;

		Component(const Component<BaseT>&) = delete;
		Component<BaseT>& operator=(const Component<BaseT>&) = delete;

		Component(Component<BaseT>&& other) noexcept;
		Component<BaseT>& operator=(Component<BaseT>&& other) noexcept;

		void clear() noexcept;
		BaseT* get_used() noexcept;
		const BaseT* get_used() const noexcept;

		template <cc::inherits_from<BaseT> DerivedT, typename... Args>
		DerivedT& add(Args&&... args);

		template <cc::inherits_from<BaseT> DerivedT, typename... Args>
		DerivedT& add_and_use(Args&&... args);

		template <cc::inherits_from<BaseT> DerivedT>
		DerivedT* use() noexcept;

		template <cc::inherits_from<BaseT> DerivedT>
		void remove() noexcept;

		template <cc::inherits_from<BaseT> DerivedT>
		DerivedT* get() noexcept;

		template <cc::inherits_from<BaseT> DerivedT>
		const DerivedT* get() const noexcept;

		template <cc::inherits_from<BaseT> DerivedT>
		bool does_exist() const noexcept;

		template <cc::inherits_from<BaseT> DerivedT>
		bool is_being_used() const noexcept;
	};
}

namespace evo
{
	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline u64_t Component<BaseT>::get_id() noexcept
	{
		static const u64_t s_value = s_next++;
		return s_value;
	}

	template <typename BaseT>
	inline Component<BaseT>::Component(Component<BaseT>&& other) noexcept
		: m_current(std::exchange(other.m_current, nullptr)), m_data(std::move(other.m_data)) {}

	template <typename BaseT>
	inline Component<BaseT>& Component<BaseT>::operator=(Component<BaseT>&& other) noexcept
	{
		m_current = std::exchange(other.m_current, nullptr);
		m_data = std::move(other.m_data);

		return *this;
	}

	template <typename BaseT>
	inline void Component<BaseT>::clear() noexcept
	{
		m_current = nullptr;
		m_data.clear();
	}

	template <typename BaseT>
	inline BaseT* Component<BaseT>::get_used() noexcept
	{
		return m_current;
	}

	template <typename BaseT>
	inline const BaseT* Component<BaseT>::get_used() const noexcept
	{
		return m_current;
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT, typename... Args>
	inline DerivedT& Component<BaseT>::add(Args&&... args)
	{
		const u64_t id = get_id<DerivedT>();
		if (m_data.size() <= id)
			m_data.resize(id + 1);

		auto& [uptr, vptr] = m_data[id];
		if (!uptr)
		{
			DerivedT* const ptr = new DerivedT(std::forward<Args>(args)...);
			vptr = reinterpret_cast<void*>(ptr);
			uptr = std::unique_ptr<BaseT>(ptr);
		}

		return *reinterpret_cast<DerivedT*>(vptr);
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT, typename... Args>
	inline DerivedT& Component<BaseT>::add_and_use(Args&&... args)
	{
		const u64_t id = get_id<DerivedT>();
		if (m_data.size() <= id)
			m_data.resize(id + 1);

		auto& [uptr, vptr] = m_data[id];
		if (!uptr)
		{
			DerivedT* const ptr = new DerivedT(std::forward<Args>(args)...);
			vptr = reinterpret_cast<void*>(ptr);
			uptr = std::unique_ptr<BaseT>(ptr);
		}

		m_current = uptr.get();
		return *reinterpret_cast<DerivedT*>(vptr);
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline DerivedT* Component<BaseT>::use() noexcept
	{
		const u64_t id = get_id<DerivedT>();
		if (m_data.size() <= id)
			return nullptr;

		auto& [uptr, vptr] = m_data[id];
		if (!uptr)
			return nullptr;

		m_current = uptr.get();
		return reinterpret_cast<DerivedT*>(vptr);
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline void Component<BaseT>::remove() noexcept
	{
		const u64_t id = get_id<DerivedT>();
		if (m_data.size() <= id)
			return;

		auto& [uptr, vptr] = m_data[id];
		if (uptr)
		{
			if (m_current == uptr.get())
				m_current = nullptr;

			uptr.reset();
			vptr = nullptr;
		}
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline DerivedT* Component<BaseT>::get() noexcept
	{
		const u64_t id = get_id<DerivedT>();
		return id < m_data.size() ? reinterpret_cast<DerivedT*>(m_data[id].second) : nullptr;
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline const DerivedT* Component<BaseT>::get() const noexcept
	{
		const u64_t id = get_id<DerivedT>();
		return id < m_data.size() ? reinterpret_cast<const DerivedT*>(m_data[id].second) : nullptr;
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline bool Component<BaseT>::does_exist() const noexcept
	{
		const u64_t id = get_id<DerivedT>();
		return id < m_data.size() && static_cast<bool>(m_data[id].first);
	}

	template <typename BaseT> template <cc::inherits_from<BaseT> DerivedT>
	inline bool Component<BaseT>::is_being_used() const noexcept
	{
		const u64_t id = get_id<DerivedT>();
		return does_exist<DerivedT>() && m_data[id].first.get() == m_current;
	}
}
