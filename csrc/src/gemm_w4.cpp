// int4 weight-only GEMM: weights are dequantized one K-block at a time into the
// fp32 panel layout and fed to the fp32 microkernel. Weight memory traffic is 8x
// lower than fp32; arithmetic is unchanged. This helps when the GEMM is
// weight-bandwidth-bound (small N, e.g. batch-1 linear layers) and does little
// when it is compute-bound (large N, e.g. early convolutions).
#include <algorithm>
#include <stdexcept>

#include "internal.h"

namespace edgeai {

void gemm_w4(Isa isa, const PackedW4& A, int N, const float* B, const float* bias, bool relu,
             float* C) {
  using namespace detail;
  const int M = A.M, K = A.K, G = A.group;
  if (G <= 0) throw std::invalid_argument("gemm_w4: group must be positive");
  const int groups_per_row = (K + G - 1) / G;
  const int bytes_per_row = (K + 1) / 2;
  if (K == 0) std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> panel(static_cast<size_t>(KC) * MR);
  for (int m0 = 0; m0 < M; m0 += MR) {
    const int mrows = std::min(MR, M - m0);
    for (int k0 = 0; k0 < K; k0 += KC) {
      const int kc = std::min(KC, K - k0);
      for (int r = 0; r < MR; ++r) {
        const int m = m0 + r;
        if (m >= M) {
          for (int k = 0; k < kc; ++k) panel[k * MR + r] = 0.0f;
          continue;
        }
        const uint8_t* q = A.q.data() + static_cast<size_t>(m) * bytes_per_row;
        const float* s = A.scales.data() + static_cast<size_t>(m) * groups_per_row;
        for (int k = 0; k < kc; ++k) {
          const int kk = k0 + k;
          const uint8_t byte = q[kk >> 1];
          const int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
          panel[k * MR + r] = static_cast<float>((nib ^ 8) - 8) * s[kk / G];
        }
      }
      f32_panel(isa, kc, panel.data(), N, B + static_cast<size_t>(k0) * N,
                C + static_cast<size_t>(m0) * N, mrows, k0 > 0);
    }
  }
  bias_act(M, N, bias, relu, C);
}

}  // namespace edgeai
