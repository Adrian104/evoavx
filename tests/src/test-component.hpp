#include "pch.hpp"
#include <evoavx/evoavx.hpp>

struct BaseA {};
struct BaseB
{
	int m_var;
	BaseB(int var) : m_var(var) {}
};

struct DerivedA0 : public BaseA {};
struct DerivedA1 : public BaseA {};
struct DerivedA2 : public BaseA {};
struct DerivedA3 : public BaseA {};
struct DerivedB0 : public BaseB
{
	DerivedB0(int var) : BaseB(var) {}
};

TEST_CASE("Component class template works as expected")
{
	evo::Component<BaseA> a;
	evo::Component<BaseB> b;

	REQUIRE_FALSE(a.does_exist<DerivedA1>());
	REQUIRE_FALSE(a.is_being_used<DerivedA1>());

	a.add_and_use<DerivedA3>();
	a.add<DerivedA0>();
	b.add_and_use<DerivedB0>(5);
	a.add_and_use<DerivedA1>();

	REQUIRE(a.get<DerivedA0>() != nullptr);
	REQUIRE(a.get<DerivedA1>() != nullptr);
	REQUIRE(a.get<DerivedA2>() == nullptr);
	REQUIRE(a.get<DerivedA3>() != nullptr);
	REQUIRE(b.get<DerivedB0>() != nullptr);

	REQUIRE(a.does_exist<DerivedA0>());
	REQUIRE(a.does_exist<DerivedA1>());
	REQUIRE_FALSE(a.does_exist<DerivedA2>());
	REQUIRE(a.does_exist<DerivedA3>());
	REQUIRE(b.does_exist<DerivedB0>());

	REQUIRE_FALSE(a.is_being_used<DerivedA0>());
	REQUIRE(a.is_being_used<DerivedA1>());
	REQUIRE_FALSE(a.is_being_used<DerivedA2>());
	REQUIRE_FALSE(a.is_being_used<DerivedA3>());
	REQUIRE(b.is_being_used<DerivedB0>());

	REQUIRE(a.get_used() != nullptr);
	REQUIRE(b.get_used() != nullptr);
	REQUIRE(b.get_used()->m_var == 5);

	a.remove<DerivedA1>();
	a.add<DerivedA2>();
	b.add_and_use<DerivedB0>(15);

	REQUIRE(a.get<DerivedA0>() != nullptr);
	REQUIRE(a.get<DerivedA1>() == nullptr);
	REQUIRE(a.get<DerivedA2>() != nullptr);
	REQUIRE(a.get<DerivedA3>() != nullptr);
	REQUIRE(b.get<DerivedB0>() != nullptr);

	REQUIRE(a.does_exist<DerivedA0>());
	REQUIRE_FALSE(a.does_exist<DerivedA1>());
	REQUIRE(a.does_exist<DerivedA2>());
	REQUIRE(a.does_exist<DerivedA3>());
	REQUIRE(b.does_exist<DerivedB0>());

	REQUIRE_FALSE(a.is_being_used<DerivedA0>());
	REQUIRE_FALSE(a.is_being_used<DerivedA1>());
	REQUIRE_FALSE(a.is_being_used<DerivedA2>());
	REQUIRE_FALSE(a.is_being_used<DerivedA3>());
	REQUIRE(b.is_being_used<DerivedB0>());

	REQUIRE(a.get_used() == nullptr);
	REQUIRE(b.get_used() != nullptr);
	REQUIRE(b.get_used()->m_var == 5);

	a.use<DerivedA0>();
	b.clear();

	REQUIRE(a.get<DerivedA0>() != nullptr);
	REQUIRE(a.get<DerivedA1>() == nullptr);
	REQUIRE(a.get<DerivedA2>() != nullptr);
	REQUIRE(a.get<DerivedA3>() != nullptr);
	REQUIRE(b.get<DerivedB0>() == nullptr);

	REQUIRE(a.does_exist<DerivedA0>());
	REQUIRE_FALSE(a.does_exist<DerivedA1>());
	REQUIRE(a.does_exist<DerivedA2>());
	REQUIRE(a.does_exist<DerivedA3>());
	REQUIRE_FALSE(b.does_exist<DerivedB0>());

	REQUIRE(a.is_being_used<DerivedA0>());
	REQUIRE_FALSE(a.is_being_used<DerivedA1>());
	REQUIRE_FALSE(a.is_being_used<DerivedA2>());
	REQUIRE_FALSE(a.is_being_used<DerivedA3>());
	REQUIRE_FALSE(b.is_being_used<DerivedB0>());

	REQUIRE(a.get_used() != nullptr);
	REQUIRE(b.get_used() == nullptr);
}
