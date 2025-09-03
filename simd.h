#pragma once
#if defined(_MSC_VER)
#include<intrin.h>
#elif defined(__GNUC__)
#include<immintrin.h>
#endif

#include<iostream>
#include<type_traits>
#include<functional>
#include<cmath>
#include"lazyEvaluation.hpp"

namespace simd
{
	class data8f {
		alignas(32) __m256 data_;
	public:
		data8f() :data_() {}
		data8f(__m256 value):data_(value){}
		operator __m256() const noexcept { return data_; }
	};

	class data4f {
		alignas(16) __m128 data_;
	public:
		data4f() :data_() {}
		data4f(__m128 value) :data_(value) {}
		operator __m128() const noexcept { return data_; }
	};

	class data4d {
		alignas(32) __m256d data_;
	public:
		data4d() :data_() {}
		data4d(__m256d value) :data_(value) {}
		operator __m256d() const  { return data_; }
	};

	template<>
	struct ScalarConverter<float,data4f> {
		constexpr static bool value = true;
		static ScalarExpr<data4f> convert(float f) { return data4f(_mm_set1_ps(f)); }
	};

	template<>
	struct ScalarConverter<float, data8f> {
		constexpr static bool value = true;
		static ScalarExpr<data8f> convert(float f) { return data8f(_mm256_set1_ps(f)); }
	};

	template<>
	struct ScalarConverter<double, data4d> {
		constexpr static bool value = true;
		static ScalarExpr<data4d> convert(double f) { return data4d(_mm256_set1_pd(f)); }
	};

	template<>
	struct ScalarConverter<double, data4f> {
		constexpr static bool value = true;
		static ScalarExpr<data4f> convert(double f) { return data4f(_mm_set1_ps((float)f)); }
	};

	template<>
	struct ScalarConverter<double, data8f> {
		constexpr static bool value = true;
		static ScalarExpr<data8f> convert(double f) { return data8f(_mm256_set1_ps((float)f)); }
	};

	template<>
	struct ScalarConverter<float, data4d> {
		constexpr static bool value = true;
		static ScalarExpr<data4d> convert(float f) { return data4d(_mm256_set1_pd((double)f)); }
	};

	data8f operator+(const data8f& lhs, const data8f& rhs)  noexcept;
	data8f operator-(const data8f& lhs, const data8f& rhs)  noexcept;
	data8f operator*(const data8f& lhs, const data8f& rhs)  noexcept;
	data8f operator/(const data8f& lhs, const data8f& rhs)  noexcept;

	data4d operator+(const data4d& lhs, const data4d& rhs)  noexcept;
	data4d operator-(const data4d& lhs, const data4d& rhs)  noexcept;
	data4d operator*(const data4d& lhs, const data4d& rhs)  noexcept;
	data4d operator/(const data4d& lhs, const data4d& rhs)  noexcept;

	data4f operator+(const data4f& lhs, const data4f& rhs)  noexcept;
	data4f operator-(const data4f& lhs, const data4f& rhs)  noexcept;
	data4f operator*(const data4f& lhs, const data4f& rhs)  noexcept;
	data4f operator/(const data4f& lhs, const data4f& rhs)  noexcept;

	template<>
	class MultiIdentity<data4d> : public Expr<MultiIdentity<data4d>, data4d> {
	public:
		MultiIdentity() {}
		data4d eval() const noexcept {
			return _mm256_set1_pd(1.);
		}
	};

	template<>
	class AddiIdentity<data4d> : public Expr<AddiIdentity<data4d>, data4d> {
	public:
		AddiIdentity() {}
		data4d eval() const noexcept {
			return _mm256_set1_pd(0.);
		}
	};

	template<>
	class MultiIdentity<data4f> : public Expr<MultiIdentity<data4f>, data4f> {
	public:
		MultiIdentity() {}
		data4f eval() const noexcept {
			return _mm_set1_ps(1.f);
		}
	};

	template<>
	class AddiIdentity<data4f> : public Expr<AddiIdentity<data4f>, data4f> {
	public:
		AddiIdentity() {}
		data4f eval() const noexcept {
			return _mm_set1_ps(0.f);
		}
	};

	template<>
	class MultiIdentity<data8f> : public Expr<MultiIdentity<data8f>, data8f> {
	public:
		MultiIdentity() {}
		data8f eval() const noexcept {
			return _mm256_set1_ps(1.f);
		}
	};

	template<>
	class AddiIdentity<data8f> : public Expr<AddiIdentity<data8f>, data8f> {
	public:
		AddiIdentity() {}
		data8f eval() const noexcept {
			return _mm256_set1_ps(0.f);
		}
	};

	class vec4f;
	class vec8f : public Expr<vec8f, data8f>
	{
		alignas(32) data8f data_;
	public:
		using type = float;
		using eval_type = data8f;
		vec8f() noexcept;
		~vec8f() = default;

		vec8f(const float*);
		vec8f(__m256 other);
		vec8f(float _1, float _2, float _3, float _4, float _5, float _6, float _7, float _8);
		vec8f(float s);

		vec8f(const vec8f& other) noexcept = default;
		vec8f(vec8f&& other) noexcept = default;

		template<class exprType>
		vec8f(const Expr<exprType, data8f>& expr) {
			data_ = static_cast<const exprType&>(expr).eval();
		}

		vec8f& operator=(const vec8f& other) noexcept = default;
		vec8f& operator=(vec8f&& other) noexcept = default;
		
		vec8f& operator+=(const vec8f& other);
		vec8f& operator-=(const vec8f& other);
		vec8f& operator*=(const vec8f& other);
		vec8f& operator/=(const vec8f& other);

		template<class Other>
		vec8f& operator+=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ + other.eval();
			return *this;
		}
		template<class Other>
		vec8f& operator-=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ - other.eval();
			return *this;
		}
		template<class Other>
		vec8f& operator*=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ * other.eval();
			return *this;
		}
		template<class Other>
		vec8f& operator/=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ / other.eval();
			return *this;
		}

		float operator[](std::size_t Index) const;


		/*template<std::size_t Index>
		float at() const{
			constexpr int lo = 0;
			constexpr int hi = 1;
			if constexpr (Index == 0) {
				return _mm256_cvtss_f32(data_);
			}
			else if constexpr (Index < 4){
				__m128 temp =  _mm256_extractf128_ps(data_, lo);
				return _mm_cvtss_f32(_mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0, 0, 0, Index)));
			}
			else if constexpr (Index < 8){
				__m128 temp = _mm256_extractf128_ps(data_, hi);
				return _mm_cvtss_f32(_mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0, 0, 0, Index - 4)));
			}
			else {
				return 0.f;
			}
		}*/

		template<std::size_t Index>
		float at() const {
			return *(reinterpret_cast<const float*>(&data_) + Index);
		}

		template<std::size_t Index>
		float safe_at() const {
			alignas(32) float temp[8];
			store(temp);
			return temp[Index];
		}

		template<std::size_t... Is>
		float _partial_sum(std::index_sequence<Is...>) const {
			return (0.f + ... + at<Is>());
		}

		template<std::size_t I>
		float partial_sum() const {
			return _partial_sum(std::make_index_sequence<I>{});
		}

		template<std::size_t... Is>
		float _partial_product(std::index_sequence<Is...>) const {
			return (1.f * ... * at<Is>());
		}

		template<std::size_t I>
		float partial_product() const {
			return _partial_product(std::make_index_sequence<I>{});
		}


		__m256 eval() const;

		vec8f& load(const float* pointer);
		const vec8f& store(float* pointer) const;

		vec4f to_vec4_lo();
		vec4f to_vec4_hi();

		operator vec4f();
		
		float sum() const;
		float sum(int num) const;

		vec8f sqrt() const;

		constexpr static int capacity() { return 8; }

	};

	class vec4d;

	class vec4f : public Expr<vec4f, data4f>
	{
		alignas(16) data4f data_;
	public:
		using type = float;
		using eval_type = data4f;
		vec4f() noexcept;
		~vec4f() = default;

		vec4f(const float*);
		vec4f(__m128 other);
		vec4f(float _1, float _2, float _3, float _4);
		vec4f(float s);

		vec4f(const vec4f& other) noexcept = default;
		vec4f(vec4f&& other) noexcept = default;
		vec4f& operator=(const vec4f& other) noexcept = default;
		vec4f& operator=(vec4f&& other) noexcept = default;

		template<class exprType>
		vec4f(const Expr<exprType, data4f>& expr) {
			data_ = static_cast<const exprType&>(expr).eval();
		}
		
		vec4f& operator+=(const vec4f& other);
		vec4f& operator-=(const vec4f& other);
		vec4f& operator*=(const vec4f& other);
		vec4f& operator/=(const vec4f& other);

		template<class Other>
		vec4f& operator+=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ + other.eval();
			return *this;
		}
		template<class Other>
		vec4f& operator-=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ - other.eval();
			return *this;
		}
		template<class Other>
		vec4f& operator*=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ * other.eval();
			return *this;
		}
		template<class Other>
		vec4f& operator/=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ / other.eval();
			return *this;
		}

		vec8f operator|(vec4f other) const;

		float operator[](std::size_t Index) const;

		operator vec4d() const;

		/*template<std::size_t Index>
		float at() const {
			return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(0, 0, 0, Index)));
		}*/

		template<std::size_t Index>
		float at() const {
			return *(reinterpret_cast<const float*>(&data_) + Index);
		}

		template<std::size_t Index>
		float safe_at() const {
			alignas(32) float temp[4];
			store(temp);
			return temp[Index];
		}

		template<std::size_t... Is>
		float _partial_sum(std::index_sequence<Is...>) const {
			return (0.f + ... + at<Is>());
		}

		template<std::size_t I>
		float partial_sum() const {
			return _partial_sum(std::make_index_sequence<I>{});
		}

		template<std::size_t... Is>
		float _partial_product(std::index_sequence<Is...>) const {
			return (1.f * ... * at<Is>());
		}

		template<std::size_t I>
		float partial_product() const {
			return _partial_product(std::make_index_sequence<I>{});
		}

		__m128 eval() const;

		vec4f& load(const float* pointer);
		const vec4f& store(float* pointer) const;

		float sum() const;
		float sum(int num) const;

		vec4f sqrt() const;


		vec8f combine(vec4f other) const;

		

		constexpr static int capacity() { return 4; }
	};

	class vec4d : public Expr<vec4d,data4d>{
		alignas(32) data4d data_;
	public:
		using type = double;
		using eval_type = data4d;
		vec4d() noexcept;
		~vec4d() = default;

		vec4d(const double*);
		vec4d(__m256d other);
		vec4d(double _1, double _2, double _3, double _4);
		vec4d(double s);

		vec4d(const vec4d& other) noexcept = default;
		vec4d(vec4d&& other) noexcept = default;
		vec4d& operator=(const vec4d& other) noexcept = default;
		vec4d& operator=(vec4d&& other) noexcept = default;

		template<class exprType>
		vec4d(const Expr<exprType, data4d>& expr) {
			data_ = static_cast<const exprType&>(expr).eval();
		}

		vec4d& operator+=(const vec4d& other);
		vec4d& operator-=(const vec4d& other);
		vec4d& operator*=(const vec4d& other);
		vec4d& operator/=(const vec4d& other);

		template<class Other>
		vec4d& operator+=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ + other.eval();
			return *this;
		}
		template<class Other>
		vec4d& operator-=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ - other.eval();
			return *this;
		}
		template<class Other>
		vec4d& operator*=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ * other.eval();
			return *this;
		}
		template<class Other>
		vec4d& operator/=(const Other& other) {
			static_assert(std::is_base_of_v<Expr<Other, eval_type>, Other>, "The type of other need to inherit from simd::Expr template");
			data_ = data_ / other.eval();
			return *this;
		}

		double operator[](std::size_t Index) const;

		operator vec4f() const;

		/*template<std::size_t Index>
		double at() const {
			constexpr int lo = 0;
			constexpr int hi = 1;
			if constexpr (Index == 0) {
				return _mm256_cvtsd_f64(data_);
			}
			else if constexpr (Index == 1) {
				return _mm256_cvtsd_f64(_mm256_unpackhi_pd(data_, data_));
			}
			else if constexpr (Index < 4) {
				if constexpr (Index == 2) {
					return _mm_cvtsd_f64(_mm256_extractf128_pd(data_, hi));
				}
				else if constexpr (Index == 3) {
					return _mm_cvtsd_f64(_mm256_extractf128_pd(_mm256_unpackhi_pd(data_, data_), hi));
				}
				return 0.;
			}
			else {
				return 0.;
			}
		}*/

		template<std::size_t Index>
		double at() const {
			return *(reinterpret_cast<const double*>(&data_) + Index);
		}

		template<std::size_t Index>
		double safe_at() const {
			alignas(64) double temp[4];
			store(temp);
			return temp[Index];
		}

		template<std::size_t... Is>
		double _partial_sum(std::index_sequence<Is...>) const {
			return (0. + ... + at<Is>());
		}

		template<std::size_t I>
		double partial_sum() const {
			return _partial_sum(std::make_index_sequence<I>{});
		}

		template<std::size_t... Is>
		double _partial_product(std::index_sequence<Is...>) const {
			return (1. * ... * at<Is>());
		}

		template<std::size_t I>
		double partial_product() const {
			return _partial_product(std::make_index_sequence<I>{});
		}

		__m256d eval() const;

		vec4d& load(const double* pointer);
		const vec4d& store(double* pointer) const;

		double sum() const;
		double sum(int num) const;

		vec4d sqrt() const;

		constexpr static int capacity() { return 4; }
	};

	template<class vecType, 
		class SFINAE = std::void_t<decltype(std::decay_t<vecType>::capacity())>,
		std::size_t I = std::decay_t<vecType>::capacity()>
	void print(vecType&& other) {
		_print(std::forward<vecType>(other), std::make_index_sequence<I>{});
	}

	template<class vecType,std::size_t... Is>
	void _print(vecType&& other, std::index_sequence<Is...>) {
		(void)std::initializer_list<int>{ (std::cout << std::forward<vecType>(other).template at<Is>() << " ",0)... };
	}

	vec8f combine(const vec4f& lo, const vec4f& hi);

	/*template<class VecType>
	float dot_product(VecType&& vec1, VecType && vec2) {
		auto vec_ = std::forward<VecType>(vec1) - std::forward<VecType>(vec2);
		return vec_.template pow<2>().sum();
	}*/

	constexpr float addf(float lhs, float rhs) {
		return lhs + rhs;
	}
	
	constexpr float subf(float lhs, float rhs) {
		return lhs - rhs;
	}

	constexpr float mulf(float lhs, float rhs) {
		return lhs * rhs;
	}

	constexpr float divf(float lhs, float rhs) {
		return lhs / rhs;
	}

	template<int I>
	struct factor {
		constexpr static unsigned int value = factor<I - 1>::value * I;
	};

	template<>
	struct factor<0> {
		constexpr static unsigned int value = 1;
	};

	template<int I>
	constexpr int factor_v = factor<I>::value;

	template<int top,int bottom>
	struct div{
		constexpr static float value = divf(top, bottom);
	};

	template<int top,int bottom>
	constexpr float div_v = div<top, bottom>::value;

	vec8f sinf(const vec8f& vec);
	vec8f cosf(const vec8f& vec);
	vec8f tanf(const vec8f& vec);

	vec4f sinf(const vec4f& vec);
	vec4f cosf(const vec4f& vec);
	vec4f tanf(const vec4f& vec);
	
	vec4d sin(const vec4d& vec);
	vec4d cos(const vec4d& vec);
	vec4d tan(const vec4d& vec);

	template<class remainType>
	struct narrow_invoke{
		template<class deleteType>
		operator deleteType() = delete;
		operator remainType() { return std::declval<remainType>(); }
	};

	template<class vecType,class = void>
	struct has_capacity : std::false_type {};

	template<class vecType>
	struct has_capacity<vecType,std::void_t<decltype(vecType::capacity())>> : std::true_type {};

	template<class vecType>
	inline constexpr bool has_capacity_v = has_capacity<vecType>::value;

	template<class ...Args>
	decltype(auto) vectorize_invoke(Args&&... args)  {	
		static_assert(sizeof...(Args) > 1, "The number of parameters need to be larger than 1");
		if constexpr (sizeof...(Args) == 2) {
			return _vectorize_invoke_unary(std::forward<Args>(args)...);
		}
		else if constexpr(sizeof...(Args) == 3) {
			return _vectorize_invoke_binary(std::forward<Args>(args)...);
		}
		else {
			return _vectorize_invoke_bulk(std::forward<Args>(args)...);
		}
	}

	template<class funcType, class vecType>
		requires requires {
		std::decay_t<vecType>::capacity();
	}
	std::decay_t<vecType> _vectorize_invoke_unary(funcType&& func, vecType&& vec) {
		
		return _vectorize_invoke_wrapper_unary(std::forward<funcType>(func), std::forward<vecType>(vec), std::make_index_sequence<std::decay_t<vecType>::capacity()>{});
		
	}

	template<class funcType, class vecType>
		requires requires {
		std::decay_t<vecType>::capacity();
	}
	std::decay_t<vecType> _vectorize_invoke_binary(funcType&& func, vecType&& vec1, vecType&& vec2) {
		return _vectorize_invoke_wrapper_binary(std::forward<funcType>(func), std::forward<vecType>(vec1), std::forward<vecType>(vec2), std::make_index_sequence<std::decay_t<vecType>::capacity()>{});
	}

	template<class funcType, class... vecTypes>
		requires requires {
		std::decay_t<typename std::tuple_element_t<0,std::tuple<vecTypes...>>>::capacity();
	}
	std::decay_t<typename std::tuple_element_t<0, std::tuple<vecTypes...>>> _vectorize_invoke_bulk(funcType&& func, vecTypes&&... vecs) {
		using containerType = std::decay_t<typename std::tuple_element_t<0, std::tuple<vecTypes...>>>;
		return _vectorize_invoke_wrapper_bulk(std::forward<funcType>(func), std::make_index_sequence<containerType::capacity()>{}, std::forward<vecTypes>(vecs)...);
	}

	//cpp 17
	/*template<class vecType, class funcType, 
		class SFINAE = std::void_t<
		std::enable_if_t<std::is_invocable_v <std::decay_t<funcType>, float > && 
		std::is_constructible_v<std::decay_t<vecType>,float*> && 
		(std::is_same_v<float,std::invoke_result_t<std::decay_t<funcType>,float>> ||
			std::is_same_v<double, std::invoke_result_t<std::decay_t<funcType>, float>>)>,
		decltype(std::declval<std::decay_t<vecType>>().store(std::add_pointer_t<float>())),
		decltype(std::declval<std::decay_t<vecType>>().load(std::add_pointer_t<float>()))
		> ,
		std::size_t... Is >
	std::decay_t<vecType> _vectorize_invoke_wrapper(vecType&& vec, funcType&& func, std::index_sequence<Is...>){
		using containerType = std::decay_t<vecType>;
		alignas(32 * std::decay_t<vecType>::capacity()) float input[std::decay_t<vecType>::capacity()];
		vec.store(input);
		alignas(32 * std::decay_t<vecType>::capacity()) float output[] = { std::invoke(std::forward<funcType>(func),input[Is])...};
		return containerType(static_cast<float*>(output));
	}*/
	
	//cpp 20
	template<class funcType, class vecType, std::size_t... Is >
		requires requires (std::decay_t<vecType> vec, typename std::decay_t<vecType>::type* p) {
		requires std::is_invocable_v<std::decay_t<funcType>, narrow_invoke<typename std::decay_t<vecType>::type>> &&
		         std::is_constructible_v<std::decay_t<vecType>, typename std::decay_t<vecType>::type*> &&
	            (std::is_same_v<std::invoke_result_t<std::decay_t<funcType>, typename std::decay_t<vecType>::type>, float> ||
				 std::is_same_v<std::invoke_result_t<std::decay_t<funcType>, typename std::decay_t<vecType>::type>, double>);
		vec.store(p);
	}
	decltype(auto) _vectorize_invoke_wrapper_unary(funcType&& func, vecType&& vec, std::index_sequence<Is...>) {
		using containerType = std::decay_t<vecType>;
		alignas(sizeof(typename containerType::type)) typename containerType::type input[containerType::capacity()];
		vec.store(input);
		alignas(sizeof(typename containerType::type)) typename containerType::type output[] = { std::invoke(std::forward<funcType>(func),input[Is])... };
		return containerType(static_cast<typename containerType::type*>(output));
	}

	template<class funcType, class vecType, std::size_t... Is >
		requires requires (std::decay_t<vecType> vec, typename std::decay_t<vecType>::type* p) {
		requires std::is_invocable_v<std::decay_t<funcType>, narrow_invoke<typename std::decay_t<vecType>::type>, narrow_invoke<typename std::decay_t<vecType>::type>>&&
	std::is_constructible_v<std::decay_t<vecType>, typename std::decay_t<vecType>::type*> &&
		(std::is_same_v<std::invoke_result_t<std::decay_t<funcType>, typename std::decay_t<vecType>::type, typename std::decay_t<vecType>::type>, float> ||
			std::is_same_v<std::invoke_result_t<std::decay_t<funcType>, typename std::decay_t<vecType>::type, typename std::decay_t<vecType>::type>, double>);
	vec.store(p);
	}
	decltype(auto) _vectorize_invoke_wrapper_binary(funcType&& func, vecType&& vec1, vecType&& vec2, std::index_sequence<Is...>) {
		using containerType = std::decay_t<vecType>;
		alignas(sizeof(typename containerType::type)) typename containerType::type input1[containerType::capacity()];
		alignas(sizeof(typename containerType::type)) typename containerType::type input2[containerType::capacity()];
		vec1.store(input1);
		vec2.store(input2);
		alignas(sizeof(typename containerType::type)) typename containerType::type output[] = { std::invoke(std::forward<funcType>(func),input1[Is],input2[Is])... };
		return containerType(static_cast<typename containerType::type*>(output));
	}

	template<std::size_t I, typename funcType, typename... vecTypes>
	decltype(auto) _apply_at_index(funcType&& func, vecTypes&&... vecs) {
		return std::invoke(std::forward<funcType>(func), std::forward<vecTypes>(vecs).template at<I>()...);
	}

	template<class funcType, class... vecTypes, std::size_t... Is >
	decltype(auto) _vectorize_invoke_wrapper_bulk(funcType&& func, std::index_sequence<Is...>, vecTypes&&... vecs) {
		using container = std::decay_t<std::tuple_element_t<0, std::tuple<vecTypes...>>>;
		using value = typename container::type;

		alignas(alignof(value)) value buffer[] = {
			_apply_at_index<Is>(std::forward<funcType>(func), std::forward<vecTypes>(vecs)...)...
		};

		return container(static_cast<value*>(buffer));
	}
}