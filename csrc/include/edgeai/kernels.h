// Edge-AI CPU kernels. All matrices are row-major and single-threaded.
//
// Convention: C[M,N] = A[M,K] * B[K,N], where A holds weights (M = output
// channels) and B holds activations (for convolution, B is the im2col matrix
// with N = output pixels).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgeai {

enum class Isa { Scalar, Neon, Avx2, Avx512 };

const char* isa_name(Isa isa);
Isa parse_isa(const std::string& name);  // "auto" -> best_isa()
bool isa_supported(Isa isa);
Isa best_isa();
std::vector<Isa> supported_isas();

// ---------------------------------------------------------------- fp32 ----
// Weights packed into 4-row panels: data[(m / 4) * K * 4 + k * 4 + (m % 4)],
// zero-padded to a multiple of 4 rows.
struct PackedF32 {
  int M = 0, K = 0;
  std::vector<float> data;
};
PackedF32 pack_f32(int M, int K, const float* A);

// C = A * B (+ bias[m]) (then ReLU if relu). bias may be null.
void gemm_f32(Isa isa, const PackedF32& A, int N, const float* B, const float* bias, bool relu,
              float* C);

// ---------------------------------------------------------------- int8 ----
// Exact int8 x int8 -> int32. Packing layout depends on the ISA.
struct PackedS8 {
  Isa isa = Isa::Scalar;
  int M = 0, K = 0, Kp = 0;
  std::vector<int8_t> data;
};
PackedS8 pack_s8(Isa isa, int M, int K, const int8_t* A);
void gemm_s8(const PackedS8& A, int N, const int8_t* B, int32_t* C);

// out[m,n] = acc[m,n] * row_scale[m] * col_scale (+ bias[m]) (then ReLU).
void requantize(int M, int N, const int32_t* acc, const float* row_scale, float col_scale,
                const float* bias, bool relu, float* out);

// ------------------------------------------------- int4 weight-only ----
// q: [M, ceil(K/2)] bytes, element k of row m in byte k/2 (low nibble = even k),
// two's-complement nibbles in [-8, 7]. scales: [M, ceil(K/group)].
struct PackedW4 {
  int M = 0, K = 0, group = 0;
  std::vector<uint8_t> q;
  std::vector<float> scales;
};
void gemm_w4(Isa isa, const PackedW4& A, int N, const float* B, const float* bias, bool relu,
             float* C);

// --------------------------------------------------- 2:4 structured ----
// For each row and each group of 4 consecutive k: 2 kept values and one meta
// byte (idx0 | idx1 << 2), idx in [0, 3]. K must be a multiple of 4.
struct Sparse24 {
  int M = 0, K = 0;
  std::vector<float> values;  // [M, K/2]
  std::vector<uint8_t> meta;  // [M, K/4]
};
void gemm_sparse24(Isa isa, const Sparse24& A, int N, const float* B, const float* bias,
                   bool relu, float* C);

// -------------------------------------------------- unstructured CSR ----
struct Csr {
  int M = 0, K = 0;
  std::vector<float> values;
  std::vector<int32_t> col;
  std::vector<int32_t> row_ptr;  // size M + 1
};
void gemm_csr(Isa isa, const Csr& A, int N, const float* B, const float* bias, bool relu,
              float* C);

// ------------------------------------------------ im2col / quantize ----
// x: [C, H, W] -> cols: [C*kh*kw, Ho*Wo] (zero padding), matching torch conv2d.
template <typename T>
void im2col(const T* x, int C, int H, int W, int kh, int kw, int sh, int sw, int ph, int pw,
            T* cols);
int conv_out_size(int in, int k, int stride, int pad);

// out = clamp(round_half_even(x / scale), -127, 127). Divides (not multiplies by
// 1/scale) so results are bit-identical to the PyTorch simulation and numpy.
void quantize_s8(const float* x, size_t n, float scale, int8_t* out);
float absmax(const float* x, size_t n);

// ------------------------------------------------------ microbenchmarks ----
double peak_fma_gflops(Isa isa, double seconds);
double peak_s8_gops(Isa isa, double seconds);
double triad_gbps(size_t bytes_per_array, int reps);

}  // namespace edgeai
