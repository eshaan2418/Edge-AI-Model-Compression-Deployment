#include <algorithm>
#include <cmath>
#include <cstring>

#include "edgeai/kernels.h"

namespace edgeai {

int conv_out_size(int in, int k, int stride, int pad) { return (in + 2 * pad - k) / stride + 1; }

template <typename T>
void im2col(const T* x, int C, int H, int W, int kh, int kw, int sh, int sw, int ph, int pw,
            T* cols) {
  const int Ho = conv_out_size(H, kh, sh, ph), Wo = conv_out_size(W, kw, sw, pw);
  const size_t HW = static_cast<size_t>(Ho) * Wo;
  for (int c = 0; c < C; ++c) {
    const T* xc = x + static_cast<size_t>(c) * H * W;
    for (int i = 0; i < kh; ++i) {
      for (int j = 0; j < kw; ++j) {
        T* row = cols + (static_cast<size_t>(c) * kh * kw + i * kw + j) * HW;
        for (int oh = 0; oh < Ho; ++oh) {
          const int ih = oh * sh - ph + i;
          T* out = row + static_cast<size_t>(oh) * Wo;
          if (ih < 0 || ih >= H) {
            std::memset(out, 0, sizeof(T) * Wo);
            continue;
          }
          const T* xr = xc + static_cast<size_t>(ih) * W;
          if (sw == 1) {
            // Contiguous run of valid columns, zero borders.
            const int lo = std::clamp(pw - j, 0, Wo), hi = std::clamp(W + pw - j, 0, Wo);
            std::memset(out, 0, sizeof(T) * lo);
            if (hi > lo) std::memcpy(out + lo, xr + lo - pw + j, sizeof(T) * (hi - lo));
            std::memset(out + std::max(hi, lo), 0, sizeof(T) * (Wo - std::max(hi, lo)));
          } else {
            for (int ow = 0; ow < Wo; ++ow) {
              const int iw = ow * sw - pw + j;
              out[ow] = (iw >= 0 && iw < W) ? xr[iw] : T(0);
            }
          }
        }
      }
    }
  }
}

template void im2col<float>(const float*, int, int, int, int, int, int, int, int, int, float*);
template void im2col<int8_t>(const int8_t*, int, int, int, int, int, int, int, int, int, int8_t*);

void quantize_s8(const float* x, size_t n, float scale, int8_t* out) {
  for (size_t i = 0; i < n; ++i) {
    // nearbyint uses the current rounding mode (round-half-even by default), like np.rint.
    const float q = std::nearbyint(x[i] / scale);
    out[i] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
  }
}

float absmax(const float* x, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
  return m;
}

}  // namespace edgeai
