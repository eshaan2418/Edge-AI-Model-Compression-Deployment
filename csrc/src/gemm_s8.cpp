// Exact int8 x int8 -> int32 GEMM.
//
// NEON: `vdotq_laneq_s32` on 4x4 interleaved panels (4 rows/cols x 4 k per 16 bytes).
// AVX2 / AVX-512: operands sign-extended to int16, then `madd_epi16` (pairwise
// int16 products summed into int32). This is exact. The common `maddubs_epi16`
// trick is not: it saturates at int16 and needs unsigned activations.
#include <algorithm>
#include <cstring>
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

using detail::MR;
using detail::round_up;

int k_multiple(Isa isa) {
  switch (isa) {
    case Isa::Neon:
      return 4;
    case Isa::Avx2:
    case Isa::Avx512:
      return 2;
    default:
      return 1;
  }
}

int panel_cols(Isa isa) { return isa == Isa::Avx512 ? 32 : 16; }

// Element (m, k) of A or 0 if out of range.
inline int8_t at(const int8_t* A, int rows, int cols, int r, int c) {
  return (r < rows && c < cols) ? A[static_cast<size_t>(r) * cols + c] : 0;
}

// Pack B (activations, [K, N]) into column panels for the given ISA.
// NEON: [N/16][Kp/4][16 cols][4 k] int8.  AVX: [N/NR][Kp/2][NR cols][2 k] int16.
std::vector<int8_t> pack_b(Isa isa, int K, int Kp, int N, const int8_t* B) {
  const int NRi = panel_cols(isa);
  const int Np = round_up(N, NRi);
  std::vector<int8_t> out;
  if (isa == Isa::Neon) {
    out.assign(static_cast<size_t>(Np) * Kp, 0);
    int8_t* p = out.data();
    for (int n0 = 0; n0 < Np; n0 += NRi)
      for (int k0 = 0; k0 < Kp; k0 += 4)
        for (int j = 0; j < NRi; ++j)
          for (int t = 0; t < 4; ++t) *p++ = at(B, K, N, k0 + t, n0 + j);
  } else {
    out.assign(static_cast<size_t>(Np) * Kp * 2, 0);
    auto* p = reinterpret_cast<int16_t*>(out.data());
    for (int n0 = 0; n0 < Np; n0 += NRi)
      for (int k0 = 0; k0 < Kp; k0 += 2)
        for (int j = 0; j < NRi; ++j)
          for (int t = 0; t < 2; ++t) *p++ = at(B, K, N, k0 + t, n0 + j);
  }
  return out;
}

#if defined(__aarch64__)
// tile[4][16] (int32) for one 4-row A panel and one 16-column B panel.
void mk_neon(int groups, const int8_t* Ap, const int8_t* Bp, int32_t* tile) {
  int32x4_t acc[4][4];
  for (auto& row : acc)
    for (auto& v : row) v = vdupq_n_s32(0);
  for (int g = 0; g < groups; ++g) {
    const int8x16_t a = vld1q_s8(Ap + 16 * g);
    const int8_t* b = Bp + 64 * g;
    const int8x16_t b0 = vld1q_s8(b), b1 = vld1q_s8(b + 16);
    const int8x16_t b2 = vld1q_s8(b + 32), b3 = vld1q_s8(b + 48);
#define EDGEAI_ROW(r)                                \
  acc[r][0] = vdotq_laneq_s32(acc[r][0], b0, a, r); \
  acc[r][1] = vdotq_laneq_s32(acc[r][1], b1, a, r); \
  acc[r][2] = vdotq_laneq_s32(acc[r][2], b2, a, r); \
  acc[r][3] = vdotq_laneq_s32(acc[r][3], b3, a, r);
    EDGEAI_ROW(0)
    EDGEAI_ROW(1)
    EDGEAI_ROW(2)
    EDGEAI_ROW(3)
#undef EDGEAI_ROW
  }
  for (int r = 0; r < 4; ++r)
    for (int j = 0; j < 4; ++j) vst1q_s32(tile + r * 16 + 4 * j, acc[r][j]);
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx2"))) void mk_avx2(int pairs, const int16_t* Ap, const int16_t* Bp,
                                             int32_t* tile) {
  __m256i acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
  for (int p = 0; p < pairs; ++p) {
    const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + 32 * p));
    const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + 32 * p + 16));
    for (int r = 0; r < MR; ++r) {
      int32_t pair;
      std::memcpy(&pair, Ap + 8 * p + 2 * r, sizeof(pair));
      const __m256i a = _mm256_set1_epi32(pair);
      acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(b0, a));
      acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(b1, a));
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile + r * 16), acc[r][0]);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile + r * 16 + 8), acc[r][1]);
  }
}

__attribute__((target("avx512f,avx512bw"))) void mk_avx512(int pairs, const int16_t* Ap,
                                                           const int16_t* Bp, int32_t* tile) {
  __m512i acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm512_setzero_si512();
  for (int p = 0; p < pairs; ++p) {
    const __m512i b0 = _mm512_loadu_si512(Bp + 64 * p);
    const __m512i b1 = _mm512_loadu_si512(Bp + 64 * p + 32);
    for (int r = 0; r < MR; ++r) {
      int32_t pair;
      std::memcpy(&pair, Ap + 8 * p + 2 * r, sizeof(pair));
      const __m512i a = _mm512_set1_epi32(pair);
      acc[r][0] = _mm512_add_epi32(acc[r][0], _mm512_madd_epi16(b0, a));
      acc[r][1] = _mm512_add_epi32(acc[r][1], _mm512_madd_epi16(b1, a));
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm512_storeu_si512(tile + r * 32, acc[r][0]);
    _mm512_storeu_si512(tile + r * 32 + 16, acc[r][1]);
  }
}
#endif

void gemm_scalar(const PackedS8& A, int N, const int8_t* B, int32_t* C) {
  const int M = A.M, K = A.K;
  for (int m = 0; m < M; ++m) {
    int32_t* c = C + static_cast<size_t>(m) * N;
    std::fill(c, c + N, 0);
    for (int k = 0; k < K; ++k) {
      const int32_t a = A.data[static_cast<size_t>(m) * K + k];
      const int8_t* b = B + static_cast<size_t>(k) * N;
      for (int n = 0; n < N; ++n) c[n] += a * b[n];
    }
  }
}

}  // namespace

PackedS8 pack_s8(Isa isa, int M, int K, const int8_t* A) {
  PackedS8 p;
  p.isa = isa;
  p.M = M;
  p.K = K;
  p.Kp = round_up(std::max(K, 1), k_multiple(isa));
  if (isa == Isa::Scalar) {
    p.data.assign(A, A + static_cast<size_t>(M) * K);
    return p;
  }
  const int Mp = round_up(M, MR);
  if (isa == Isa::Neon) {
    // [M/4][Kp/4][4 rows][4 k]
    p.data.assign(static_cast<size_t>(Mp) * p.Kp, 0);
    int8_t* q = p.data.data();
    for (int m0 = 0; m0 < Mp; m0 += MR)
      for (int k0 = 0; k0 < p.Kp; k0 += 4)
        for (int r = 0; r < MR; ++r)
          for (int t = 0; t < 4; ++t) *q++ = at(A, M, K, m0 + r, k0 + t);
  } else {
    // [M/4][Kp/2][4 rows][2 k] as int16
    p.data.assign(static_cast<size_t>(Mp) * p.Kp * 2, 0);
    auto* q = reinterpret_cast<int16_t*>(p.data.data());
    for (int m0 = 0; m0 < Mp; m0 += MR)
      for (int k0 = 0; k0 < p.Kp; k0 += 2)
        for (int r = 0; r < MR; ++r)
          for (int t = 0; t < 2; ++t) *q++ = at(A, M, K, m0 + r, k0 + t);
  }
  return p;
}

void gemm_s8(const PackedS8& A, int N, const int8_t* B, int32_t* C) {
  if (A.isa == Isa::Scalar) return gemm_scalar(A, N, B, C);
  const int M = A.M, Kp = A.Kp, NRi = panel_cols(A.isa);
  const std::vector<int8_t> Bp = pack_b(A.isa, A.K, Kp, N, B);
  std::vector<int32_t> tile(static_cast<size_t>(MR) * NRi);
  const size_t a_panel_bytes = static_cast<size_t>(MR) * Kp * (A.isa == Isa::Neon ? 1 : 2);
  const size_t b_panel_bytes = static_cast<size_t>(NRi) * Kp * (A.isa == Isa::Neon ? 1 : 2);
  for (int m0 = 0; m0 < M; m0 += MR) {
    const int8_t* ap = A.data.data() + (m0 / MR) * a_panel_bytes;
    const int mrows = std::min(MR, M - m0);
    for (int n0 = 0; n0 < N; n0 += NRi) {
      const int8_t* bp = Bp.data() + (n0 / NRi) * b_panel_bytes;
      switch (A.isa) {
#if defined(__aarch64__)
        case Isa::Neon:
          mk_neon(Kp / 4, ap, bp, tile.data());
          break;
#endif
#if defined(__x86_64__)
        case Isa::Avx2:
          mk_avx2(Kp / 2, reinterpret_cast<const int16_t*>(ap),
                  reinterpret_cast<const int16_t*>(bp), tile.data());
          break;
        case Isa::Avx512:
          mk_avx512(Kp / 2, reinterpret_cast<const int16_t*>(ap),
                    reinterpret_cast<const int16_t*>(bp), tile.data());
          break;
#endif
        default:
          throw std::logic_error("gemm_s8: ISA not compiled into this build");
      }
      const int nc = std::min(NRi, N - n0);
      for (int r = 0; r < mrows; ++r) {
        std::memcpy(C + static_cast<size_t>(m0 + r) * N + n0, tile.data() + r * NRi,
                    sizeof(int32_t) * nc);
      }
    }
  }
}

void requantize(int M, int N, const int32_t* acc, const float* row_scale, float col_scale,
                const float* bias, bool relu, float* out) {
  for (int m = 0; m < M; ++m) {
    const float s = row_scale[m] * col_scale;
    const float b = bias ? bias[m] : 0.0f;
    const int32_t* a = acc + static_cast<size_t>(m) * N;
    float* o = out + static_cast<size_t>(m) * N;
    for (int n = 0; n < N; ++n) {
      const float v = static_cast<float>(a[n]) * s + b;
      o[n] = relu ? std::max(v, 0.0f) : v;
    }
  }
}

}  // namespace edgeai
