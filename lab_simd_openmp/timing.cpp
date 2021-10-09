#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include "timing.hpp"

#define K 10
#define MAX_RUNS 200
#define MIN_TEST_CYCLES (50L * 1000L * 1000L)
#define TOLERANCE_NUMERATOR 1005
#define TOLERANCE_DENOMINATOR 1000


static uint64_t measure_once_intrinsic(intrinsic_function_type f, size_t n) {
    uint32_t start_cycles_high, start_cycles_low;
    uint32_t end_cycles_high, end_cycles_low;
    __asm__ volatile(
    "cpuid\n\t"
    "rdtsc\n\t"
    :
    "=d" (start_cycles_high), "=a" (start_cycles_low)
    ::"%rbx", "%rcx"
    );
    f(n);
    __asm__ volatile(
    "rdtscp\n\t"
    "mov %%edx, %0\n\t"
    "mov %%eax, %1\n\t"
    "cpuid\n\t"
    :
    "=r" (end_cycles_high), "=r" (end_cycles_low)
    ::"%rax", "%rbx", "%rcx", "%rdx"
    );
    uint64_t start_cycles = ((uint64_t) start_cycles_high) << 32 | start_cycles_low;
    uint64_t end_cycles = ((uint64_t) end_cycles_high) << 32 | end_cycles_low;
    return end_cycles - start_cycles;
}

static uint64_t measure_once_integrate(integrate_function_type f, int n, double a_bound, double b_bound, arithmetic_function_type func) {
    uint32_t start_cycles_high, start_cycles_low;
    uint32_t end_cycles_high, end_cycles_low;
    __asm__ volatile(
    "cpuid\n\t"
    "rdtsc\n\t"
    :
    "=d" (start_cycles_high), "=a" (start_cycles_low)
    ::"%rbx", "%rcx"
    );
    f(n, a_bound, b_bound, func);
    __asm__ volatile(
    "rdtscp\n\t"
    "mov %%edx, %0\n\t"
    "mov %%eax, %1\n\t"
    "cpuid\n\t"
    :
    "=r" (end_cycles_high), "=r" (end_cycles_low)
    ::"%rax", "%rbx", "%rcx", "%rdx"
    );
    uint64_t start_cycles = ((uint64_t) start_cycles_high) << 32 | start_cycles_low;
    uint64_t end_cycles = ((uint64_t) end_cycles_high) << 32 | end_cycles_low;
    return end_cycles - start_cycles;
}

uint64_t measure_function_integrate(integrate_function_type f, int n, double a_bound, double b_bound, arithmetic_function_type func) {
    const uint64_t huge_measure = UINT64_MAX / (TOLERANCE_DENOMINATOR * TOLERANCE_NUMERATOR);
    uint64_t measures[K];
    std::fill(std::begin(measures), std::begin(measures) + K, huge_measure);
    int num_runs = 0;
    uint64_t total_cycles = 0;
    while (measures[K - 1] >= huge_measure ||
           measures[0] * TOLERANCE_NUMERATOR / TOLERANCE_DENOMINATOR < measures[K - 1]) {
        uint64_t cur_measure = measure_once_integrate(f, n, a_bound, b_bound, func);
        total_cycles += cur_measure;
        int insertion_point = K;
        for (int i = 0; i < K; ++i) {
            if (cur_measure < measures[i]) {
                insertion_point = i;
                break;
            }
        }
        if (insertion_point < K) {
            memmove(&measures[insertion_point + 1], &measures[insertion_point],
                    (K - insertion_point - 1) * sizeof(*measures));
            measures[insertion_point] = cur_measure;
        }
        if (num_runs++ > MAX_RUNS && total_cycles > MIN_TEST_CYCLES) {
            break;
        }
    }
    return measures[K / 2 - 1];
}

uint64_t measure_function_intrinsic(intrinsic_function_type f, size_t n) {
    const uint64_t huge_measure = UINT64_MAX / (TOLERANCE_DENOMINATOR * TOLERANCE_NUMERATOR);
    uint64_t measures[K];
    std::fill(std::begin(measures), std::begin(measures) + K, huge_measure);
    int num_runs = 0;
    uint64_t total_cycles = 0;
    while (measures[K - 1] >= huge_measure ||
           measures[0] * TOLERANCE_NUMERATOR / TOLERANCE_DENOMINATOR < measures[K - 1]) {
        uint64_t cur_measure = measure_once_intrinsic(f, n);
        total_cycles += cur_measure;
        int insertion_point = K;
        for (int i = 0; i < K; ++i) {
            if (cur_measure < measures[i]) {
                insertion_point = i;
                break;
            }
        }
        if (insertion_point < K) {
            memmove(&measures[insertion_point + 1], &measures[insertion_point],
                    (K - insertion_point - 1) * sizeof(*measures));
            measures[insertion_point] = cur_measure;
        }
        if (num_runs++ > MAX_RUNS && total_cycles > MIN_TEST_CYCLES) {
            break;
        }
    }
    return measures[K / 2 - 1];
}
