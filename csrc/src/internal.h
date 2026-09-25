// Shared internals: fp32 register-tile microkernel reused by gemm_f32 and gemm_w4.
#pragma once

#include "edgeai/kernels.h"

namespace edgeai::detail {

constexpr int MR = 4;    // rows per register tile / weight panel
constexpr int NR = 16;   // columns per register tile
constexpr int KC = 256;  // K block: a 4 x 256 fp32 weight panel is 4 KiB (L1-resident)

// C rows [0, mrows) of a 4-row block (+)= Ap[kc][4] * B[kc, N] (row-major, ld = N).
void f32_panel(Isa isa, int kc, const float* Ap, int N, const float* B, float* C, int mrows,
               bool accumulate);

// C[m, :] = act(C[m, :] + bias[m]) for m < M.
void bias_act(int M, int N, const float* bias, bool relu, float* C);

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

}  // namespace edgeai::detail
