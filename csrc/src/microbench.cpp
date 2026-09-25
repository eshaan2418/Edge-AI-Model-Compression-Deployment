// Single-core machine peaks for roofline analysis: independent FMA / dot-product
// chains (enough accumulators to cover latency) and a STREAM-style triad.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "edgeai/kernels.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace edgeai {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_s(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

volatile float g_sink_f;
volatile int32_t g_sink_i;

constexpr int kInner = 1 << 16;

#if defined(__aarch64__)
double fma_neon(double seconds) {
  float32x4_t acc[12];
  for (auto& a : acc) a = vdupq_n_f32(0.0f);
  const float32x4_t x = vdupq_n_f32(1.0001f), y = vdupq_n_f32(0.9999f);
  double flops = 0;
  const auto t0 = Clock::now();
  do {
    for (int i = 0; i < kInner; ++i)
      for (auto& a : acc) a = vfmaq_f32(a, x, y);
    flops += 12.0 * 4 * 2 * kInner;
  } while (elapsed_s(t0) < seconds);
  const double t = elapsed_s(t0);
  float s = 0;
  for (auto& a : acc) s += vaddvq_f32(a);
  g_sink_f = s;
  return flops / t / 1e9;
}

double sdot_neon(double seconds) {
  int32x4_t acc[12];
  for (auto& a : acc) a = vdupq_n_s32(0);
  const int8x16_t x = vdupq_n_s8(1), y = vdupq_n_s8(-1);
  double ops = 0;
  const auto t0 = Clock::now();
  do {
    for (int i = 0; i < kInner; ++i)
      for (auto& a : acc) a = vdotq_s32(a, x, y);
    ops += 12.0 * 16 * 2 * kInner;  // 16 multiply-adds per instruction
  } while (elapsed_s(t0) < seconds);
  const double t = elapsed_s(t0);
  int32_t s = 0;
  for (auto& a : acc) s += vaddvq_s32(a);
  g_sink_i = s;
  return ops / t / 1e9;
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) double fma_avx2(double seconds) {
  __m256 acc[12];
  for (auto& a : acc) a = _mm256_setzero_ps();
  const __m256 x = _mm256_set1_ps(1.0001f), y = _mm256_set1_ps(0.9999f);
  double flops = 0;
  const auto t0 = Clock::now();
  do {
    for (int i = 0; i < kInner; ++i)
      for (auto& a : acc) a = _mm256_fmadd_ps(x, y, a);
    flops += 12.0 * 8 * 2 * kInner;
  } while (elapsed_s(t0) < seconds);
  const double t = elapsed_s(t0);
  alignas(32) float out[8];
  __m256 s = _mm256_setzero_ps();
  for (auto& a : acc) s = _mm256_add_ps(s, a);
  _mm256_store_ps(out, s);
  g_sink_f = out[0];
  return flops / t / 1e9;
}

__attribute__((target("avx2"))) double madd_avx2(double seconds) {
  __m256i acc[12];
  for (auto& a : acc) a = _mm256_setzero_si256();
  const __m256i x = _mm256_set1_epi16(1), y = _mm256_set1_epi16(-1);
  double ops = 0;
  const auto t0 = Clock::now();
  do {
    for (int i = 0; i < kInner; ++i)
      for (auto& a : acc) a = _mm256_add_epi32(a, _mm256_madd_epi16(x, y));
    ops += 12.0 * 16 * 2 * kInner;
  } while (elapsed_s(t0) < seconds);
  const double t = elapsed_s(t0);
  alignas(32) int32_t out[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(out), acc[0]);
  g_sink_i = out[0];
  return ops / t / 1e9;
}
#endif

double fma_scalar(double seconds) {
  float acc[8] = {};
  double flops = 0;
  const auto t0 = Clock::now();
  do {
    for (int i = 0; i < kInner; ++i)
      for (auto& a : acc) a = a * 0.9999f + 1.0001f;
    flops += 8.0 * 2 * kInner;
  } while (elapsed_s(t0) < seconds);
  const double t = elapsed_s(t0);
  g_sink_f = acc[0] + acc[7];
  return flops / t / 1e9;
}

}  // namespace

double peak_fma_gflops(Isa isa, double seconds) {
  switch (isa) {
#if defined(__aarch64__)
    case Isa::Neon:
      return fma_neon(seconds);
#endif
#if defined(__x86_64__)
    case Isa::Avx2:
    case Isa::Avx512:
      return fma_avx2(seconds);
#endif
    default:
      return fma_scalar(seconds);
  }
}

double peak_s8_gops(Isa isa, double seconds) {
  switch (isa) {
#if defined(__aarch64__)
    case Isa::Neon:
      return sdot_neon(seconds);
#endif
#if defined(__x86_64__)
    case Isa::Avx2:
    case Isa::Avx512:
      return madd_avx2(seconds);
#endif
    default:
      return fma_scalar(seconds);
  }
}

double triad_gbps(size_t bytes_per_array, int reps) {
  const size_t n = bytes_per_array / sizeof(float);
  std::vector<float> a(n, 0.0f), b(n, 1.0f), c(n, 2.0f);
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + 3.0f * c[i];
    best = std::min(best, elapsed_s(t0));
  }
  g_sink_f = a[n / 2];
  return 3.0 * n * sizeof(float) / best / 1e9;  // 2 reads + 1 write per element
}

}  // namespace edgeai
