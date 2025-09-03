#include"simd.h"

namespace simd
{
	// vec8f
	vec8f::vec8f() noexcept : data_() {

	}

	vec8f::vec8f(const float* pointer) {
		data_ = _mm256_loadu_ps(pointer);
	}

	vec8f::vec8f(__m256 other) {
		data_ = other;
	}

	vec8f::vec8f(float _1, float _2, float _3, float _4, float _5, float _6, float _7, float _8) {
		data_ = _mm256_setr_ps(_1, _2, _3, _4, _5, _6, _7, _8);
	}

	vec8f::vec8f(float s) {
		data_ = _mm256_set1_ps(s);
	}

	vec8f& vec8f::operator+=(const vec8f& other) {
		data_ = _mm256_add_ps(data_, other.data_);
		return *this;
	}

	vec8f& vec8f::operator-=(const vec8f& other) {
		data_ = _mm256_sub_ps(data_, other.data_);
		return *this;
	}

	vec8f& vec8f::operator*=(const vec8f& other) {
		data_ = _mm256_mul_ps(data_, other.data_);
		return *this;
	}

	vec8f& vec8f::operator/=(const vec8f& other) {
		data_ = _mm256_div_ps(data_, other.data_);
		return *this;
	}

	float vec8f::operator[](std::size_t Index) const {
		switch (Index)
		{
		case 0:return at<0>();
		case 1:return at<1>();
		case 2:return at<2>();
		case 3:return at<3>();
		case 4:return at<4>();
		case 5:return at<5>();
		case 6:return at<6>();
		case 7:return at<7>();
		default:
			return 0.f;
		}
	}

	__m256 vec8f::eval() const {
		return data_;
	}

	vec8f& vec8f::load(const float* pointer) {
		data_ = _mm256_loadu_ps(pointer);
		return *this;
	}

	const vec8f& vec8f::store(float* pointer) const {
		_mm256_storeu_ps(pointer, data_);
		return *this;
	}

	vec4f vec8f::to_vec4_lo() {
		return _mm256_extractf128_ps(data_, 0);
	}

	vec4f vec8f::to_vec4_hi() {
		return _mm256_extractf128_ps(data_, 1);
	}

	vec8f::operator vec4f() {
		return to_vec4_lo();
	}

	float vec8f::sum() const {
		__m128 low = _mm256_extractf128_ps(data_, 0);
		__m128 high = _mm256_extractf128_ps(data_, 1);

		low = _mm_hadd_ps(low, low);
		low = _mm_hadd_ps(low, low);

		high = _mm_hadd_ps(high, high);
		high = _mm_hadd_ps(high, high);

		return _mm_cvtss_f32(low) + _mm_cvtss_f32(high);
	}

	float vec8f::sum(int num) const {
		switch (num)
		{
		case 1:return partial_sum<1>();
		case 2:return partial_sum<2>();
		case 3:return partial_sum<3>();
		case 4:return partial_sum<4>();
		case 5:return partial_sum<5>();
		case 6:return partial_sum<6>();
		case 7:return partial_sum<7>();
		case 8:return sum();
		default:
			return 0.f;
		}
	}

	vec8f vec8f::sqrt() const {
		return _mm256_sqrt_ps(data_);
	}


	// vec4f
	vec4f::vec4f() noexcept : data_() {

	}

	vec4f::vec4f(const float* pointer) {
		data_ = _mm_loadu_ps(pointer);
	}

	vec4f::vec4f(__m128 other) {
		data_ = other;
	}

	vec4f::vec4f(float _1, float _2, float _3, float _4) {
		data_ = _mm_setr_ps(_1, _2, _3, _4);
	}

	vec4f::vec4f(float s) {
		data_ = _mm_set1_ps(s);
	}

	vec4f& vec4f::operator+=(const vec4f& other) {
		data_ = _mm_add_ps(data_, other.data_);
		return *this;
	}

	vec4f& vec4f::operator-=(const vec4f& other) {
		data_ = _mm_sub_ps(data_, other.data_);
		return *this;
	}

	vec4f& vec4f::operator*=(const vec4f& other) {
		data_ = _mm_mul_ps(data_, other.data_);
		return *this;
	}

	vec4f& vec4f::operator/=(const vec4f& other) {
		data_ = _mm_div_ps(data_, other.data_);
		return *this;
	}

	float vec4f::operator[](std::size_t Index) const{
		switch (Index)
		{
		case 0:return at<0>();
		case 1:return at<1>();
		case 2:return at<2>();
		case 3:return at<3>();
		default:
			return 0.f;
		}
	}

	vec4f::operator vec4d() const {
		__m128 low_f = data_;

		__m128 high_f = _mm_movehl_ps(data_, data_);

		__m128d low_d = _mm_cvtps_pd(low_f);               // f0, f1 ¡ú d0, d1
		__m128d high_d = _mm_cvtps_pd(high_f);             // f2, f3 ¡ú d2, d3

		return _mm256_insertf128_pd(_mm256_castpd128_pd256(low_d), high_d, 1);
	}

	__m128 vec4f::eval() const {
		return data_;
	}

	vec4f& vec4f::load(const float* pointer) {
		data_ = _mm_loadu_ps(pointer);
		return *this;
	}

	const vec4f& vec4f::store(float* pointer) const {
		_mm_storeu_ps(pointer, data_);
		return *this;
	}

	float vec4f::sum() const {
		__m128 temp = _mm_hadd_ps(data_, data_);
		temp = _mm_hadd_ps(temp, temp);
		return _mm_cvtss_f32(temp);
	}
	
	float vec4f::sum(int num) const {
		switch (num)
		{
		case 1:return partial_sum<1>();
		case 2:return partial_sum<2>();
		case 3:return partial_sum<3>();
		case 4:return sum();
		default:
			return 0.f;
		}
	}

	vec4f vec4f::sqrt() const {
		return _mm_sqrt_ps(data_);
	}

	vec8f vec4f::combine(vec4f other) const {
		return _mm256_set_m128(other.data_, data_);
	}

	vec8f vec4f::operator|(vec4f other) const{
		return combine(other);
	}

	vec8f combine(const vec4f& lo, const vec4f& hi) {
		return _mm256_set_m128(hi.eval(), lo.eval());
	}

	template<class T>
	struct sin_helper {
		T operator()(T f) noexcept {
			return std::sin(f);
		}
	};

	template<class T>
	struct cos_helper {
		T operator()(T f) noexcept {
			return std::cos(f);
		}
	};

	template<class T>
	struct tan_helper {
		T operator()(T f) noexcept {
			return std::tan(f);
		}
	};


	vec8f sinf(const vec8f& vec) {
		//x - 1/3! * x**3 + 1/5! * x**5 - 1/7! * x**7 + 1/9! * x**9
		//vec8f output = vec;
		//auto vec2 = vec.pow<2>();
		//auto vecfactor = vec * vec2;//vec3
		//output -= div_t<1, factor_t<3>> * vecfactor;
		//vecfactor *= vec2;//vec5
		//output += div_t<1, factor_t<5>>* vecfactor;
		//vecfactor *= vec2; //vec7
		//output -= div_t<1, factor_t<7>>* vecfactor;
		//vecfactor *= vec2;//vec9
		//output += div_t<1, factor_t<9>>* vecfactor;
		//return output;
		return vectorize_invoke(sin_helper<float>{},vec);
	}

	vec8f cosf(const vec8f& vec) {
		// 1 - 1/2! * x**2 + 1/4! * x**4 - 1/6! * x**6 + 1/8! * x**8 - 1/10! * x**10
		//vec8f output = vec8f(1.f);
		//auto vec2 = vec.pow<2>();
		//auto vecfactor = vec2;//vec2
		//output -= div_t<1, factor_t<2>>* vecfactor;
		//vecfactor *= vec2;//vec4f
		//output += div_t<1, factor_t<4>>* vecfactor;
		//vecfactor *= vec2; //vec6
		//output -= div_t<1, factor_t<6>> * vecfactor;
		//vecfactor *= vec2;//vec8f
		//output += div_t<1, factor_t<8>>* vecfactor;
		//vecfactor *= vec2;//vec10
		//output -= div_t<1, factor_t<10>>* vecfactor;
		return vectorize_invoke(cos_helper<float>{}, vec);
	}

	vec8f tanf(const vec8f& vec) {
		return vectorize_invoke(tan_helper<float>{}, vec);
	}

	vec4f sinf(const vec4f& vec) {
		return vectorize_invoke(sin_helper<float>{}, vec);
	}

	vec4f cosf(const vec4f& vec) {
		return vectorize_invoke(cos_helper<float>{}, vec);
	}

	vec4f tanf(const vec4f& vec) {
		return vectorize_invoke(tan_helper<float>{}, vec);
	}

	vec4d sin(const vec4d& vec) {
		return vectorize_invoke(sin_helper<double>{}, vec);
	}

	vec4d cos(const vec4d& vec) {
		return vectorize_invoke(cos_helper<double>{}, vec);
	}

	vec4d tan(const vec4d& vec) {
		return vectorize_invoke(tan_helper<double>{}, vec);
	}

	vec4d::vec4d() noexcept : data_(){
	}

	vec4d::vec4d(const double* pointer){
		data_ = _mm256_loadu_pd(pointer);
	}

	vec4d::vec4d(__m256d other){
		data_ = other;
	}

	vec4d::vec4d(double _1, double _2, double _3, double _4){
		data_ = _mm256_setr_pd(_1, _2, _3, _4);
	}
	vec4d::vec4d(double s){
		data_ = _mm256_set1_pd(s);
	}

	//vec4d vec4d::operator+(const vec4d& other) const{
	//	return _mm256_add_pd(data_, other.data_);
	//}

	vec4d& vec4d::operator+=(const vec4d& other){
		data_ = _mm256_add_pd(data_, other.data_);
		return *this;
	}

	vec4d& vec4d::operator-=(const vec4d& other){
		data_ = _mm256_sub_pd(data_, other.data_);
		return *this;
	}

	vec4d& vec4d::operator*=(const vec4d& other) {
		data_ = _mm256_mul_pd(data_, other.data_);
		return *this;
	}

	vec4d& vec4d::operator/=(const vec4d& other) {
		data_ = _mm256_div_pd(data_, other.data_);
		return *this;
	}

	double vec4d::operator[](std::size_t Index) const{
		switch (Index)
		{
		case 0: return at<0>();
		case 1: return at<1>();
		case 2: return at<2>();
		case 3: return at<3>();
		default:
			return 0.;
		}
	}

	vec4d::operator vec4f() const{
		__m128d low = _mm256_castpd256_pd128(data_);           // v[0], v[1]
		__m128d high = _mm256_extractf128_pd(data_, 1);         // v[2], v[3]

		__m128 low_f = _mm_cvtpd_ps(low);                  // ¡ú f0, f1, x, x
		__m128 high_f = _mm_cvtpd_ps(high);                // ¡ú f2, f3, x, x

		return _mm_movelh_ps(low_f, high_f);
	}

	__m256d vec4d::eval() const{
		return data_;
	}

	vec4d& vec4d::load(const double* pointer) {
		data_ = _mm256_loadu_pd(pointer);
		return *this;
	}

	const vec4d& vec4d::store(double* pointer) const {
		_mm256_storeu_pd(pointer, data_);
		return *this;
	}

	double vec4d::sum() const {
		__m256d hadd = _mm256_hadd_pd(data_, data_);
		 hadd = _mm256_hadd_pd(hadd, hadd);// hadd = [a+b, c+d, a+b, c+d]
		double sum = _mm256_cvtsd_f64(hadd);  // result[0] contains the sum
		return sum;
	}

	double vec4d::sum(int num) const{
		switch (num)
		{
		case 1: return partial_sum<1>();
		case 2: return partial_sum<2>();
		case 3: return partial_sum<3>();
		case 4: return sum();
		default:
			return 0.;
		}
	}

	vec4d vec4d::sqrt() const{
		return _mm256_sqrt_pd(data_);
	}

	data8f operator+(const data8f& lhs, const data8f& rhs)  noexcept {
		return _mm256_add_ps(lhs, rhs);
	}

	data8f operator-(const data8f& lhs, const data8f& rhs)  noexcept {
		return _mm256_sub_ps(lhs, rhs);
	}

	data8f operator*(const data8f& lhs, const data8f& rhs)  noexcept {
		return _mm256_mul_ps(lhs, rhs);
	}

	data8f operator/(const data8f& lhs, const data8f& rhs)  noexcept {
		return _mm256_div_ps(lhs, rhs);
	}

	data4d operator+(const data4d& lhs, const data4d& rhs)  noexcept {
		return _mm256_add_pd(lhs, rhs);
	}

	data4d operator-(const data4d& lhs, const data4d& rhs)  noexcept {
		return _mm256_sub_pd(lhs, rhs);
	}

	data4d operator*(const data4d& lhs, const data4d& rhs)  noexcept {
		return _mm256_mul_pd(lhs, rhs);
	}

	data4d operator/(const data4d& lhs, const data4d& rhs)  noexcept {
		return _mm256_div_pd(lhs, rhs);
	}

	data4f operator+(const data4f& lhs, const data4f& rhs)  noexcept {
		return _mm_add_ps(lhs, rhs);
	}

	data4f operator-(const data4f& lhs, const data4f& rhs)  noexcept {
		return _mm_sub_ps(lhs, rhs);
	}

	data4f operator*(const data4f& lhs, const data4f& rhs)  noexcept {
		return _mm_mul_ps(lhs, rhs);
	}

	data4f operator/(const data4f& lhs, const data4f& rhs)  noexcept {
		return _mm_div_ps(lhs, rhs);
	}

}