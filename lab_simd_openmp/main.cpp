#include <iostream>
#include <cstdio>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <xmmintrin.h>
#include "omp.h"
#include "timing.hpp"


std::vector<double> a(MAX_SIZE);
std::vector<double> b(MAX_SIZE);


void vectorized_sum_sse(size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        __m128d x = _mm_loadu_pd(&a[i]);
        __m128d y = _mm_loadu_pd(&b[i]);
        x = _mm_add_pd(x, y);
        _mm_store_pd(&a[i], x);
    }
}

void vectorized_sum_avx2(size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        __m256d x = _mm256_loadu_pd(&a[i]);
        __m256d y = _mm256_loadu_pd(&b[i]);
        x = _mm256_add_pd(x, y);
        _mm256_store_pd(&a[i], x);
    }
}


void loop_vectorization_sum(size_t n) {
    for (size_t i = 0; i < n; i += 1) {
        a[i] += b[i];
    }
}


double numeric_integration(int n, double a_bound, double b_bound, arithmetic_function_type f) {
    double _sum = 0.0;
    const double dx = (b_bound - a_bound) / n;
    for (int i = 0; i < n; ++i) {
        _sum += dx * f(dx * i);
    }
    return _sum;
}

double numeric_integration_omp(int n, double a_bound, double b_bound, arithmetic_function_type f) {
    double _sum = 0.0;
    double dx = (b_bound - a_bound) / n;
    #pragma omp parallel for reduction(+:_sum)
    for (int i = 0; i < n; ++i) {
        _sum += dx * f(dx * i);
    }
    return _sum;
}

double numeric_integration_simd(int n, double a_bound, double b_bound, arithmetic_function_type f) {
    double _sum = 0.0;
    double dx = (b_bound - a_bound) / n;
    #pragma omp simd share(_sum) reduction(+:_sum)
    for (int i = 0; i < n; ++i) {
        _sum += dx * f(dx * i);
    }
    return _sum;
}


double exp_x2(double x) {
    // \int_0^1 e^{x^2} = \frac{\sqrt{\pi}}{2}erfi(1) \approx 1.46265
    return exp(pow(x, 2));
}


int main() {
    std::cout << "sse: " << __builtin_cpu_supports("sse") << "\n";
    std::cout << "sse2: " << __builtin_cpu_supports("sse2") << "\n";
    std::cout << "avx: " << __builtin_cpu_supports("avx") << "\n";
    std::cout << "avx2: " << __builtin_cpu_supports("avx2") << "\n";
    std::cout << "avx512f: " << __builtin_cpu_supports("avx512f") << "\n";


    std::fill(std::begin(a), std::begin(a) + MAX_SIZE, (double) 3e3);
    std::fill(std::begin(b), std::begin(b) + MAX_SIZE, (double) -1.123e3);

    std::cout << "\n++++++++++++++++++++++++++++++++++++++++\n";
    printf("%-10s Cycles/Element for %s IPC    T\n", "Size", "Loop Sum");
    for (size_t i = 16; i <= MAX_SIZE; i *= 2) {
        uint64_t t = measure_function_intrinsic(loop_vectorization_sum, i);
        double cpe = (double) t / (double) i;
        double ipc = 1 / cpe;
        double T = cpe * i;
        printf("%-10d %7.2f \t\t\t\t\t%7.2f %7.2f\n", (int) i, cpe, ipc, T);
    }
    std::cout << "\n++++++++++++++++++++++++++++++++++++++++\n";
    printf("%-10s Cycles/Element for %s IPC    T\n", "Size", "Vec Sum SSE2");
    for (size_t i = 16; i <= MAX_SIZE; i *= 2) {
        uint64_t t = measure_function_intrinsic(vectorized_sum_sse, i);
        double cpe = (double) t / (double) i;
        double ipc = 1 / cpe;
        double T = cpe * i;
        printf("%-10d %7.2f \t\t\t\t\t%7.2f %7.2f\n", (int) i, cpe, ipc, T);
    }

    std::cout << "\n++++++++++++++++++++++++++++++++++++++++\n";
    printf("%-10s Cycles/Element for %s IPC    T\n", "Size", "Vec Sum AVX2");
    for (size_t i = 16; i <= MAX_SIZE; i *= 2) {
        uint64_t t = measure_function_intrinsic(vectorized_sum_avx2, i);
        double cpe = (double) t / (double) i;
        double ipc = 1 / cpe;
        double T = cpe * i;
        printf("%-10d %7.2f \t\t\t\t\t%7.2f %7.2f\n", (int) i, cpe, ipc, T);
    }

    int max_iter = 1000000;
    std::cout << "\n++++++++++++++++++++++++++++++++++++++++\n";
    printf("%-10s Cycles/Element for %s IPC    T\n", "Size", "One Thread Integration");
    for (size_t i = 10; i <= max_iter; i *= 10) {
        uint64_t t = measure_function_integrate(numeric_integration, i, 0, 1, exp_x2);
        double cpe = (double) t / (double) i;
        double ipc = 1 / cpe;
        double T = cpe * i;
        printf("%-10d %7.2f \t\t\t\t\t%7.2f %7.2f\n", (int) i, cpe, ipc, T);
    }

    std::cout << "\n++++++++++++++++++++++++++++++++++++++++\n";
    printf("%-10s Cycles/Element for %s IPC    T\n", "Size", "Parallel Integration");
    for (int i = 10; i <= max_iter; i *= 10) {
        uint64_t t = measure_function_integrate(numeric_integration_omp, i, 0, 1, exp_x2);
        double cpe = (double) t / (double) i;
        double ipc = 1 / cpe;
        double T = cpe * i;
        printf("%-10d %7.2f \t\t\t\t\t%7.2f %7.2f\n", (int) i, cpe, ipc, T);
    }

    std::cout << "\n++++++++++++++++++++++++++++++++++++++++\n";
    printf("%-10s Cycles/Element for %s IPC    T\n", "Size", "SIMD Integration");
    for (int i = 10; i <= max_iter; i *= 10) {
        uint64_t t = measure_function_integrate(numeric_integration_simd, i, 0, 1, exp_x2);
        double cpe = (double) t / (double) i;
        double ipc = 1 / cpe;
        double T = cpe * i;
        printf("%-10d %7.2f \t\t\t\t\t%7.2f %7.2f\n", (int) i, cpe, ipc, T);
    }
    return 0;
}
