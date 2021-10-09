#ifndef TIMIMG_HPP
#define TIMIMG_HPP

#include <cstdint>
#include <cstdio>
#include <functional>

#define MAX_SIZE (1024 * 1024)
#define K 10
#define MAX_RUNS 200
#define MIN_TEST_CYCLES (50L * 1000L * 1000L)
#define TOLERANCE_NUMERATOR 1005
#define TOLERANCE_DENOMINATOR 1000

using intrinsic_function_type = std::function<void(size_t)>;
using arithmetic_function_type = std::function<double(double)>;
using integrate_function_type = std::function<double(int, double, double, arithmetic_function_type)>;

uint64_t measure_function_intrinsic(intrinsic_function_type, size_t);

uint64_t measure_function_integrate(integrate_function_type, int, double, double, arithmetic_function_type);

#endif

