#include <algorithm>
#include <cstring>

#include "internal.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace edgeai {
namespace detail {
namespace {

// tile[4][16] = sum_k Ap[k][r] * B[k][j]
void mk_scalar(int kc, const float* Ap, const float* B, int ldb, float* tile) {
  float acc[MR * NR] = {};
  for (int k = 0; k < kc; ++k) {
    const float* b = B + static_cast<size_t>(k) * ldb;
    for (int r = 0; r < MR; ++r) {
      const float a = Ap[k * MR + r];
      for (int j = 0; j < NR; ++j) acc[r * NR + j] += a * b[j];
    }
  }
  std::memcpy(tile, acc, sizeof(acc));
}

#if defined(__aarch64__)
void mk_neon(int kc, const float* Ap, const float* B, int ldb, float* tile) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c02 = c00, c03 = c00;
  float32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  float32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00;
  float32x4_t c30 = c00, c31 = c00, c32 = c00, c33 = c00;
  for (int k = 0; k < kc; ++k) {
    const float32x4_t a = vld1q_f32(Ap + k * MR);
    const float* b = B + static_cast<size_t>(k) * ldb;
    const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4);
    const float32x4_t b2 = vld1q_f32(b + 8), b3 = vld1q_f32(b + 12);
    c00 = vfmaq_laneq_f32(c00, b0, a, 0);
    c01 = vfmaq_laneq_f32(c01, b1, a, 0);
    c02 = vfmaq_laneq_f32(c02, b2, a, 0);
    c03 = vfmaq_laneq_f32(c03, b3, a, 0);
    c10 = vfmaq_laneq_f32(c10, b0, a, 1);
    c11 = vfmaq_laneq_f32(c11, b1, a, 1);
    c12 = vfmaq_laneq_f32(c12, b2, a, 1);
    c13 = vfmaq_laneq_f32(c13, b3, a, 1);
    c20 = vfmaq_laneq_f32(c20, b0, a, 2);
    c21 = vfmaq_laneq_f32(c21, b1, a, 2);
    c22 = vfmaq_laneq_f32(c22, b2, a, 2);
    c23 = vfmaq_laneq_f32(c23, b3, a, 2);
    c30 = vfmaq_laneq_f32(c30, b0, a, 3);
    c31 = vfmaq_laneq_f32(c31, b1, a, 3);
    c32 = vfmaq_laneq_f32(c32, b2, a, 3);
    c33 = vfmaq_laneq_f32(c33, b3, a, 3);
  }
  float32x4_t cs[16] = {c00, c01, c02, c03, c10, c11, c12, c13,
                        c20, c21, c22, c23, c30, c31, c32, c33};
  for (int i = 0; i < 16; ++i) vst1q_f32(tile + 4 * i, cs[i]);
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) void mk_avx2(int kc, const float* Ap, const float* B, int ldb,
                                                 float* tile) {
  __m256 c[MR][2];
  for (int r = 0; r < MR; ++r) c[r][0] = c[r][1] = _mm256_setzero_ps();
  for (int k = 0; k < kc; ++k) {
    const float* b = B + static_cast<size_t>(k) * ldb;
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8);
    for (int r = 0; r < MR; ++r) {
      const __m256 a = _mm256_broadcast_ss(Ap + k * MR + r);
      c[r][0] = _mm256_fmadd_ps(a, b0, c[r][0]);
      c[r][1] = _mm256_fmadd_ps(a, b1, c[r][1]);
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm256_storeu_ps(tile + r * NR, c[r][0]);
    _mm256_storeu_ps(tile + r * NR + 8, c[r][1]);
  }
}
#endif

void microkernel(Isa isa, int kc, const float* Ap, const float* B, int ldb, float* tile) {
  switch (isa) {
#if defined(__aarch64__)
    case Isa::Neon:
      return mk_neon(kc, Ap, B, ldb, tile);
#endif
#if defined(__x86_64__)
    case Isa::Avx2:
    case Isa::Avx512:  // fp32 uses the AVX2 tile on AVX-512 machines too
      return mk_avx2(kc, Ap, B, ldb, tile);
#endif
    default:
      return mk_scalar(kc, Ap, B, ldb, tile);
  }
}

}  // namespace

void f32_panel(Isa isa, int kc, const float* Ap, int N, const float* B, float* C, int mrows,
               bool accumulate) {
  float tile[MR * NR];
  std::vector<float> edge;  // zero-padded copy of the last partial column panel
  for (int n0 = 0; n0 < N; n0 += NR) {
    const int nc = std::min(NR, N - n0);
    if (nc == NR) {
      microkernel(isa, kc, Ap, B + n0, N, tile);
    } else {
      edge.assign(static_cast<size_t>(kc) * NR, 0.0f);
      for (int k = 0; k < kc; ++k) {
        std::memcpy(&edge[static_cast<size_t>(k) * NR], B + static_cast<size_t>(k) * N + n0,
                    sizeof(float) * nc);
      }
      microkernel(isa, kc, Ap, edge.data(), NR, tile);
    }
    for (int r = 0; r < mrows; ++r) {
      float* c = C + static_cast<size_t>(r) * N + n0;
      const float* t = tile + r * NR;
      if (accumulate) {
        for (int j = 0; j < nc; ++j) c[j] += t[j];
      } else {
        std::memcpy(c, t, sizeof(float) * nc);
      }
    }
  }
}

void bias_act(int M, int N, const float* bias, bool relu, float* C) {
  if (!bias && !relu) return;
  for (int m = 0; m < M; ++m) {
    float* c = C + static_cast<size_t>(m) * N;
    const float b = bias ? bias[m] : 0.0f;
    for (int n = 0; n < N; ++n) {
      const float v = c[n] + b;
      c[n] = relu ? std::max(v, 0.0f) : v;
    }
  }
}

}  // namespace detail

PackedF32 pack_f32(int M, int K, const float* A) {
  PackedF32 p;
  p.M = M;
  p.K = K;
  const int Mp = detail::round_up(M, detail::MR);
  p.data.assign(static_cast<size_t>(Mp) * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    float* panel = p.data.data() + static_cast<size_t>(m / detail::MR) * K * detail::MR;
    for (int k = 0; k < K; ++k) panel[k * detail::MR + m % detail::MR] = A[static_cast<size_t>(m) * K + k];
  }
  return p;
}

void gemm_f32(Isa isa, const PackedF32& A, int N, const float* B, const float* bias, bool relu,
              float* C) {
  using namespace detail;
  const int M = A.M, K = A.K;
  if (K == 0) {
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
  }
  for (int m0 = 0; m0 < M; m0 += MR) {
    const int mrows = std::min(MR, M - m0);
    const float* panel = A.data.data() + static_cast<size_t>(m0) * K;
    float* c = C + static_cast<size_t>(m0) * N;
    for (int k0 = 0; k0 < K; k0 += KC) {
      const int kc = std::min(KC, K - k0);
      f32_panel(isa, kc, panel + static_cast<size_t>(k0) * MR, N, B + static_cast<size_t>(k0) * N,
                c, mrows, k0 > 0);
    }
  }
  bias_act(M, N, bias, relu, C);
}

}  // namespace edgeai
