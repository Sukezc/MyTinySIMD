#pragma once
#include<type_traits>
#include<functional>

namespace simd {

	template<class Derived, class EvalType>
	class Expr {
	public:
		using eval_type = EvalType;
		template<class... Args>
		EvalType eval(Args&&... args) const noexcept {
			return static_cast<const Derived&>(*this).eval(std::forward<Args>(args)...);
		}
	};

	template<class EvalType>
	class ScalarExpr : public Expr<ScalarExpr<EvalType>, EvalType> {
		EvalType value;
	public:
		using eval_type = EvalType;
		ScalarExpr(const EvalType& _value) : value(_value) { }
		ScalarExpr() : value() {}
		template<class... Args>
		EvalType eval(Args&&... args) const noexcept {
			return value;
		}

		decltype(auto) set(const EvalType& _value) {
			value = _value;
			return *this;
		}
	};

	template<class From, class To>
	struct ScalarConverter {
		constexpr static bool value = false;
	};

	template<class From, class To>
	constexpr static bool has_convert_v = ScalarConverter<From, To>::value;

	template<class T>
	concept HasEvalType = requires {typename std::decay_t<T>::eval_type; };

	template<class T>
	concept HasExprBase = HasEvalType<T> && std::is_base_of_v<Expr<std::decay_t<T>, typename std::decay_t<T>::eval_type>, std::decay_t<T>>;

	template<class T,class... Args>
	concept IsExpr = HasExprBase<T> && requires (T expr, Args&&... args) { expr.eval(std::forward<Args>(args)...); };


#define BASE_BIEXPR_DEFINE(name,op)\
template<class EvalType,class LHS, class RHS>\
	requires HasExprBase<LHS> && HasExprBase<RHS>\
	class name : public Expr<name<EvalType, LHS, RHS>, EvalType> {\
		LHS lhs;\
		RHS rhs;\
	public:\
		using eval_type = EvalType;\
		template<class Lhs,class Rhs>\
		name(Lhs&& _lhs, Rhs&&  _rhs) : lhs(std::forward<Lhs>(_lhs)), rhs(std::forward<Rhs>(_rhs)) { }\
		template<class... Args>\
		EvalType eval(Args&&... args) const noexcept {\
			return lhs.eval(std::forward<Args>(args)...) op rhs.eval(std::forward<Args>(args)...);\
		}\
	};\
	template <class Lhs,class Rhs>\
	name(Lhs&&,Rhs&&) -> name<\
		typename std::decay_t<Lhs>::eval_type,\
		std::conditional_t<std::is_lvalue_reference_v<Lhs&&>,Lhs&&,std::remove_reference_t<Lhs>>,\
		std::conditional_t<std::is_lvalue_reference_v<Rhs&&>,Rhs&&,std::remove_reference_t<Rhs>>\
	>;
	
#define BASE_BIEXPR_SCALAR_DEFINE(name,op) \
template<class EvalType,class LHS>\
	requires HasExprBase<LHS>\
	class name<EvalType,LHS,ScalarExpr<EvalType>>: public Expr<name<EvalType,LHS,ScalarExpr<EvalType>>, EvalType> {\
		LHS lhs;\
		const ScalarExpr<EvalType> rhs;\
	public:\
		using eval_type = EvalType;\
		template<class Lhs>\
		name(Lhs&& _lhs, const ScalarExpr<EvalType>& _rhs) : lhs(std::forward<Lhs>(_lhs)), rhs(_rhs) { }\
		template<class Lhs>\
		name(Lhs&& _lhs, ScalarExpr<EvalType>&& _rhs) : lhs(std::forward<Lhs>(_lhs)), rhs(std::move(_rhs)) { }\
		template<class... Args>	\
		EvalType eval(Args&&... args) const noexcept {\
			return lhs.eval(std::forward<Args>(args)...) op rhs.eval(std::forward<Args>(args)...);\
		}\
	};\
	template <class Lhs>\
	name(Lhs&&,const ScalarExpr<typename std::decay_t<Lhs>::eval_type>&) -> name<\
		typename std::decay_t<Lhs>::eval_type,\
		std::conditional_t<std::is_lvalue_reference_v<Lhs&&>, Lhs&&, std::remove_reference_t<Lhs>>,\
		ScalarExpr<typename std::decay_t<Lhs>::eval_type>\
	>;\
template <class Lhs>\
	name(Lhs&&,ScalarExpr<typename std::decay_t<Lhs>::eval_type>&&) -> name<\
		typename std::decay_t<Lhs>::eval_type,\
		std::conditional_t<std::is_lvalue_reference_v<Lhs&&>, Lhs&&, std::remove_reference_t<Lhs>>,\
		ScalarExpr<typename std::decay_t<Lhs>::eval_type>\
	>;\
template<class EvalType,class RHS>\
	requires HasExprBase<RHS>\
	class name<EvalType,ScalarExpr<EvalType>,RHS>: public Expr<name<EvalType,ScalarExpr<EvalType>,RHS>, EvalType> {\
		const ScalarExpr<EvalType> lhs;\
		RHS rhs;\
	public:\
		using eval_type = EvalType;\
		template<class Rhs>\
		name(const ScalarExpr<EvalType>& _lhs, Rhs&& _rhs) : lhs(_lhs), rhs(std::forward<Rhs>(_rhs)) { }\
		template<class Rhs>\
		name(ScalarExpr<EvalType>&& _lhs, Rhs&& _rhs) : lhs(std::move(_lhs)), rhs(std::forward<Rhs>(_rhs)) { }\
		template<class... Args>	\
		EvalType eval(Args&&... args) const noexcept {\
			return lhs.eval(std::forward<Args>(args)...) op rhs.eval(std::forward<Args>(args)...);\
		}\
	};\
	template <class Rhs>\
	name(const ScalarExpr<typename std::decay_t<Rhs>::eval_type>&, Rhs&&) -> name<\
		typename std::decay_t<Rhs>::eval_type,\
		ScalarExpr<typename std::decay_t<Rhs>::eval_type>,\
		std::conditional_t<std::is_lvalue_reference_v<Rhs&&>,Rhs&&,std::remove_reference_t<Rhs>>\
	>;\
	template <class Rhs>\
	name(ScalarExpr<typename std::decay_t<Rhs>::eval_type>&&, Rhs&&) -> name<\
		typename std::decay_t<Rhs>::eval_type,\
		ScalarExpr<typename std::decay_t<Rhs>::eval_type>,\
		std::conditional_t<std::is_lvalue_reference_v<Rhs&&>,Rhs&&,std::remove_reference_t<Rhs>>\
	>;


#define CROSS_BIEXPR_DEFINE(name,op)\
	template<class LHS,class RHS>\
	requires HasExprBase<LHS> && HasExprBase<RHS> && std::is_same_v<typename std::decay_t<LHS>::eval_type,typename std::decay_t<RHS>::eval_type>\
	decltype(auto) operator op (LHS&& lhs, RHS&& rhs) {\
		return name(std::forward<LHS>(lhs), std::forward<RHS>(rhs));\
	}


#define CROSS_BIEXPR_WIDE_DEFINE(name,op)\
	template<class LHS,class RHS>\
	requires HasExprBase<LHS> && (!HasEvalType<RHS>)\
	decltype(auto) operator op (LHS&& lhs, RHS&& rhs) {\
	static_assert(has_convert_v<std::decay_t<RHS>,typename std::decay_t<LHS>::eval_type>,"No pre-defined variable has_convert_v<From,To>");\
		return name(std::forward<LHS>(lhs),ScalarConverter<std::decay_t<RHS>,typename std::decay_t<LHS>::eval_type>::convert(rhs));\
	}\
template<class LHS,class RHS>\
	requires HasExprBase<RHS> && (!HasEvalType<LHS>)\
	decltype(auto) operator op (LHS&& lhs, RHS&& rhs) {\
	static_assert(has_convert_v<std::decay_t<LHS>,typename std::decay_t<RHS>::eval_type>,"No pre-defined variable has_convert_v<From,To>");\
		return name(ScalarConverter<std::decay_t<LHS>,typename std::decay_t<RHS>::eval_type>::convert(lhs),std::forward<RHS>(rhs));\
	}

#define BIEXPR_DEFINE(name,op)\
BASE_BIEXPR_DEFINE(name,op)\
BASE_BIEXPR_SCALAR_DEFINE(name,op)\
CROSS_BIEXPR_DEFINE(name,op)\
CROSS_BIEXPR_WIDE_DEFINE(name,op)

BIEXPR_DEFINE(AddExpr, +)
BIEXPR_DEFINE(SubExpr, -)
BIEXPR_DEFINE(MulExpr, *)
BIEXPR_DEFINE(DivExpr, /)

	template<class EvalType>
	class MultiIdentity : public Expr<MultiIdentity<EvalType>, EvalType> {
	public:
		MultiIdentity(){}
		template<class ... Args>
		EvalType eval(Args&&... args) const noexcept {
			static_assert(!std::is_same_v<EvalType,EvalType>, "The default MultiIdentity::eval is not allowed");
			return std::declval<EvalType>();
		}
	};

	template<class EvalType>
	class AddiIdentity : public Expr<AddiIdentity<EvalType>, EvalType> {
	public:
		AddiIdentity() {}
		template<class ... Args>
		EvalType eval(Args&&... args) const noexcept {
			static_assert(!std::is_same_v<EvalType, EvalType>, "The default AddiIdentity::eval is not allowed");
			return std::declval<EvalType>();
		}
	};

	template<int Exponent, class EvalType, class LHS>
		requires HasExprBase<LHS>
	class PowExpr : public Expr<PowExpr<Exponent, EvalType, LHS>, EvalType> {
		LHS lhs;
	public:
		template<class Lhs>
		PowExpr(Lhs&& lhs_) : lhs(std::forward<Lhs>(lhs_)) {}
		template<class ... Args>
		EvalType eval(Args&&... args) const noexcept {
			if constexpr (Exponent == 1) {
				return lhs.eval(std::forward<Args>(args)...);
			}
			else if constexpr (Exponent == 0) {
				return MultiIdentity<EvalType>{}.eval(std::forward<Args>(args)...);
			}
			else if constexpr (Exponent > 0) {
				if constexpr ((Exponent & 1) == 0) {
					// LHS -> std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>
					auto output = PowExpr<
						Exponent / 2, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{ lhs }.eval(std::forward<Args>(args)...);
					return output * output;
				}
				else {
					return PowExpr<
						Exponent - 1, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{lhs}.eval(std::forward<Args>(args)...) * lhs.eval(std::forward<Args>(args)...);
				}
			}
			else if constexpr (Exponent < 0) {
				if constexpr ((Exponent & 1) == 0) {
					auto output = PowExpr<
						Exponent / 2, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{ lhs }.eval(std::forward<Args>(args)...);
					return output * output;
				}
				else {
					return PowExpr<
						Exponent + 1, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{lhs}.eval(std::forward<Args>(args)...) / lhs.eval(std::forward<Args>(args)...);
				}
			}
		}

	};

	template<int Exponent, class EvalType, class LHS>
		requires HasExprBase<LHS>
	class ApowExpr : public Expr<ApowExpr<Exponent, EvalType, LHS>, EvalType> {
		LHS lhs;
	public:
		template<class Lhs>
		ApowExpr(Lhs&& lhs_) : lhs(std::forward<Lhs>(lhs_)) {}
		template<class ... Args>
		EvalType eval(Args&&... args) const noexcept {
			if constexpr (Exponent == 1) {
				return lhs.eval(std::forward<Args>(args)...);
			}
			else if constexpr (Exponent == 0) {
				return AddiIdentity<EvalType>{}.eval(std::forward<Args>(args)...);
			}
			else if constexpr (Exponent > 0) {
				if constexpr ((Exponent & 1) == 0) {
					auto output = ApowExpr<
						Exponent / 2, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{ lhs }.eval(std::forward<Args>(args)...);
					return output + output;
				}
				else {
					return ApowExpr<
						Exponent - 1, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{lhs}.eval(std::forward<Args>(args)...) + lhs.eval(std::forward<Args>(args)...);
				}
			}
			else if constexpr (Exponent < 0) {
				if constexpr ((Exponent & 1) == 0) {
					auto output = ApowExpr<
						Exponent / 2, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{ lhs }.eval(std::forward<Args>(args)...);
					return output + output;
				}
				else {
					return ApowExpr<
						Exponent + 1, 
						EvalType, 
						std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<LHS>>>>
					{lhs}.eval(std::forward<Args>(args)...) - lhs.eval(std::forward<Args>(args)...);
				}
			}
		}
	};


	template<int Exponent, class ExprIdentity>
	decltype(auto) pow(ExprIdentity&& expr) noexcept {
		return PowExpr<Exponent, 
			typename std::decay_t<ExprIdentity>::eval_type, 
			std::conditional_t<
			std::is_lvalue_reference_v<ExprIdentity&&>, 
			ExprIdentity&&, 
			std::remove_reference_t<ExprIdentity>>>
		{std::forward<ExprIdentity>(expr)};
	}

	template<int Exponent, class ExprIdentity>
	decltype(auto) apow(ExprIdentity&& expr) noexcept {
		return ApowExpr<Exponent,
			typename std::decay_t<ExprIdentity>::eval_type,
			std::conditional_t<
			std::is_lvalue_reference_v<ExprIdentity&&>,
			ExprIdentity&&,
			std::remove_reference_t<ExprIdentity>>>
		{std::forward<ExprIdentity>(expr)};
	}

	
}

