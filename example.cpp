#define _CRT_SECURE_NO_WARNINGS 1
#include"simd.h"
#include<iomanip>
#include<cmath>

void test1() {
	//声明两个向量
	auto a = simd::vec4d(1., 2., 3., 4.);
	auto b = simd::vec4d(2., -1., 7., 2.);
	std::cout << "    a: 1 2 3 4\n    b: 2 -1 7 2\n";

	std::cout << "a + b: ";
	simd::print(simd::vec4d(a + b));
	std::cout << '\n';

	std::cout << "a - b: ";
	simd::print(simd::vec4d(a - b));
	std::cout << '\n';

	std::cout << "a * b: ";
	simd::print(simd::vec4d(a * b));
	std::cout << '\n';

	std::cout << "a / b: ";
	simd::print(simd::vec4d(a / b));
	std::cout << '\n';

	std::cout << "a * 2: ";
	simd::print(simd::vec4d(a * 2.));
	std::cout << '\n';
}

void test2() {
	//声明两个向量
	double a_mem[4] = { 1.,2.,3.,4. };
	double b_mem[4] = { 2.,-1.,7.,2. };

	auto a = simd::vec4d(a_mem);
	auto b = simd::vec4d(b_mem);
	
	// d, e, f 均为表达式
	auto d = a + b - 1.;
	auto e = a - b;
	auto f = b + 1.;
	std::cout << "d = a + b - 1: ";
	simd::print(simd::vec4d(d));
	std::cout << '\n';

	std::cout << "e = a - b:    ";
	simd::print(simd::vec4d(e));
	std::cout << '\n';

	std::cout << "f = b + 1.:   ";
	simd::print(simd::vec4d(f));
	std::cout << '\n';

	std::cout << "求取d，e，f每一位的最大值；即g = {max(d[i],e[i],f[i]) for i in range(simd::vec4d::capacity())}\n";
	std::cout << "g: max(d,e,f): ";
	auto g = simd::vectorize_invoke([](double _1, double _2, double _3) {return std::max({ _1,_2,_3 }); }, simd::vec4d(d), simd::vec4d(e), simd::vec4d(f));
	simd::print(g);
	std::cout << "\n";
}

void test3() {
	auto a = simd::vec4d(1., 2., 3., 4.);
	auto b = simd::vec4d(2., -1., 7., 2.);

	auto h = simd::pow<2>(a + b);
	std::cout << "h = (a + b) ** 2\n";
	std::cout << "h: ";
	simd::print(simd::vec4d(h));
	std::cout << "\n";

	auto i = simd::apow<2>(a + b);
	std::cout << "i = (a + b) + (a + b)\n";
	std::cout << "i: ";
	simd::print(simd::vec4d(i));
	std::cout << "\n";
}

void test4() {
	auto a = simd::vec4d(1., 2., 3., 4.);
	auto b = simd::vec4d(2., -1., 7., 2.);
	
	std::cout << "sin(a):";
	simd::print(simd::sin(a));
	std::cout << "\n";
	std::cout << "sin(b):";
	simd::print(simd::sin(b));
	std::cout << "\n";

}


int main()
{
	test1();
	test2();
	test3();
	test4();
	return 0;
}

