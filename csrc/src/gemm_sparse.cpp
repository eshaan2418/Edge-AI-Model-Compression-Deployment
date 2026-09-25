// Sparse-weight GEMMs: C[m, :] = sum over nonzeros (v, k) of v * B[k, :].
//
// Each nonzero costs one row-segment load of B per FMA vector, versus the dense
// microkernel's one load per four FMAs (the A value is broadcast across a 4x16
// tile). So halving the FLOPs does not halve the time: sparse kernels are
// load-bound. That is the central effect in the sparsity study.
#include <algorithm>
#include <stdexcept>

#include "internal.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace edgeai {
namespace {

// Visitor-style: nz(i) yields (value, k) for nonzero i of the current row.
struct RowNz {
  const float* values;
  const int32_t* ks;  // CSR columns (null for 2:4)
  const uint8_t* meta;  // 2:4 metadata (null for CSR)
  int count;
  inline void get(int i, float* v, int* k) const {
    *v = values[i];
    if (ks) {
      *k = ks[i];
    } else {
      const uint8_t mb = meta[i >> 1];
      *k = 4 * (i >> 1) + ((i & 1) ? ((mb >> 2) & 3) : (mb & 3));
    }
  }
};

void row_scalar(const RowNz& row, int N, int n0, const float* B, float init, float* c) {
  for (int n = n0; n < N; ++n) c[n] = init;
  for (int i = 0; i < row.count; ++i) {
    float v;
    int k;
    row.get(i, &v, &k);
    const float* b = B + static_cast<size_t>(k) * N;
    for (int n = n0; n < N; ++n) c[n] += v * b[n];
  }
}

#if defined(__aarch64__)
int row_neon(const RowNz& row, int N, const float* B, float init, float* c) {
  int n0 = 0;
  for (; n0 + 16 <= N; n0 += 16) {
    float32x4_t a0 = vdupq_n_f32(init), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < row.count; ++i) {
      float v;
      int k;
      row.get(i, &v, &k);
      const float* b = B + static_cast<size_t>(k) * N + n0;
      a0 = vfmaq_n_f32(a0, vld1q_f32(b), v);
      a1 = vfmaq_n_f32(a1, vld1q_f32(b + 4), v);
      a2 = vfmaq_n_f32(a2, vld1q_f32(b + 8), v);
      a3 = vfmaq_n_f32(a3, vld1q_f32(b + 12), v);
    }
    vst1q_f32(c + n0, a0);
    vst1q_f32(c + n0 + 4, a1);
    vst1q_f32(c + n0 + 8, a2);
    vst1q_f32(c + n0 + 12, a3);
  }
  return n0;
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) int row_avx2(const RowNz& row, int N, const float* B,
                                                 float init, float* c) {
  int n0 = 0;
  for (; n0 + 16 <= N; n0 += 16) {
    __m256 a0 = _mm256_set1_ps(init), a1 = a0;
    for (int i = 0; i < row.count; ++i) {
      float v;
      int k;
      row.get(i, &v, &k);
      const float* b = B + static_cast<size_t>(k) * N + n0;
      const __m256 vv = _mm256_set1_ps(v);
      a0 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(b), a0);
      a1 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(b + 8), a1);
    }
    _mm256_storeu_ps(c + n0, a0);
    _mm256_storeu_ps(c + n0 + 8, a1);
  }
  return n0;
}
#endif

void run_row(Isa isa, const RowNz& row, int N, const float* B, float bias, bool relu, float* c) {
  int done = 0;
  switch (isa) {
#if defined(__aarch64__)
    case Isa::Neon:
      done = row_neon(row, N, B, bias, c);
      break;
#endif
#if defined(__x86_64__)
    case Isa::Avx2:
    case Isa::Avx512:
      done = row_avx2(row, N, B, bias, c);
      break;
#endif
    default:
      break;
  }
  row_scalar(row, N, done, B, bias, c);
  if (relu)
    for (int n = 0; n < N; ++n) c[n] = std::max(c[n], 0.0f);
}

}  // namespace

void gemm_sparse24(Isa isa, const Sparse24& A, int N, const float* B, const float* bias,
                   bool relu, float* C) {
  if (A.K % 4 != 0) throw std::invalid_argument("gemm_sparse24: K must be a multiple of 4");
  const int per_row = A.K / 2;
  for (int m = 0; m < A.M; ++m) {
    const RowNz row{A.values.data() + static_cast<size_t>(m) * per_row, nullptr,
                    A.meta.data() + static_cast<size_t>(m) * (A.K / 4), per_row};
    run_row(isa, row, N, B, bias ? bias[m] : 0.0f, relu, C + static_cast<size_t>(m) * N);
  }
}

void gemm_csr(Isa isa, const Csr& A, int N, const float* B, const float* bias, bool relu,
              float* C) {
  for (int m = 0; m < A.M; ++m) {
    const int start = A.row_ptr[m], end = A.row_ptr[m + 1];
    const RowNz row{A.values.data() + start, A.col.data() + start, nullptr, end - start};
    run_row(isa, row, N, B, bias ? bias[m] : 0.0f, relu, C + static_cast<size_t>(m) * N);
  }
}

}  // namespace edgeai
