// Python bindings. Every array argument is `noconvert`: callers must pass
// C-contiguous arrays of the exact dtype (the Python wrappers in
// edge_ai_compression/inference/kernels.py do the conversion explicitly).
// Outputs are written in place.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>

#include "edgeai/kernels.h"

namespace nb = nanobind;
using namespace edgeai;

template <typename T, size_t D>
using In = nb::ndarray<const T, nb::ndim<D>, nb::c_contig, nb::device::cpu>;
template <typename T, size_t D>
using Out = nb::ndarray<T, nb::ndim<D>, nb::c_contig, nb::device::cpu>;

namespace {

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

int dim(size_t v) { return static_cast<int>(v); }

template <typename A>
const float* opt_bias(const std::optional<In<float, 1>>& bias, const A& packed) {
  if (!bias) return nullptr;
  require(dim(bias->shape(0)) == packed.M, "bias must have shape [M]");
  return bias->data();
}

void check_out(size_t rows, size_t cols, int M, int N) {
  require(dim(rows) == M && dim(cols) == N, "out must have shape [M, N]");
}

std::vector<std::string> isa_names(const std::vector<Isa>& isas) {
  std::vector<std::string> out;
  for (Isa i : isas) out.emplace_back(isa_name(i));
  return out;
}

}  // namespace

NB_MODULE(_C, m) {
  m.doc() = "Edge-AI single-threaded CPU kernels (see csrc/include/edgeai/kernels.h)";

  m.def("supported_isas", [] { return isa_names(supported_isas()); });
  m.def("best_isa", [] { return std::string(isa_name(best_isa())); });

  // fp32
  nb::class_<PackedF32>(m, "PackedF32")
      .def_ro("M", &PackedF32::M)
      .def_ro("K", &PackedF32::K)
      .def_prop_ro("nbytes", [](const PackedF32& p) { return p.data.size() * sizeof(float); });
  m.def(
      "pack_f32", [](In<float, 2> a) { return pack_f32(dim(a.shape(0)), dim(a.shape(1)), a.data()); },
      nb::arg("a").noconvert());
  m.def(
      "gemm_f32",
      [](const std::string& isa, const PackedF32& a, In<float, 2> b,
         std::optional<In<float, 1>> bias, bool relu, Out<float, 2> out) {
        require(dim(b.shape(0)) == a.K, "b must have shape [K, N]");
        const int N = dim(b.shape(1));
        check_out(out.shape(0), out.shape(1), a.M, N);
        gemm_f32(parse_isa(isa), a, N, b.data(), opt_bias(bias, a), relu, out.data());
      },
      nb::arg("isa"), nb::arg("a"), nb::arg("b").noconvert(), nb::arg("bias").noconvert().none(),
      nb::arg("relu"), nb::arg("out").noconvert());

  // int8
  nb::class_<PackedS8>(m, "PackedS8")
      .def_ro("M", &PackedS8::M)
      .def_ro("K", &PackedS8::K)
      .def_prop_ro("isa", [](const PackedS8& p) { return std::string(isa_name(p.isa)); })
      .def_prop_ro("nbytes", [](const PackedS8& p) { return p.data.size(); });
  m.def(
      "pack_s8",
      [](const std::string& isa, In<int8_t, 2> a) {
        return pack_s8(parse_isa(isa), dim(a.shape(0)), dim(a.shape(1)), a.data());
      },
      nb::arg("isa"), nb::arg("a").noconvert());
  m.def(
      "gemm_s8",
      [](const PackedS8& a, In<int8_t, 2> b, Out<int32_t, 2> out) {
        require(dim(b.shape(0)) == a.K, "b must have shape [K, N]");
        const int N = dim(b.shape(1));
        check_out(out.shape(0), out.shape(1), a.M, N);
        gemm_s8(a, N, b.data(), out.data());
      },
      nb::arg("a"), nb::arg("b").noconvert(), nb::arg("out").noconvert());
  m.def(
      "requantize",
      [](In<int32_t, 2> acc, In<float, 1> row_scale, float col_scale,
         std::optional<In<float, 1>> bias, bool relu, Out<float, 2> out) {
        const int M = dim(acc.shape(0)), N = dim(acc.shape(1));
        require(dim(row_scale.shape(0)) == M, "row_scale must have shape [M]");
        require(!bias || dim(bias->shape(0)) == M, "bias must have shape [M]");
        check_out(out.shape(0), out.shape(1), M, N);
        requantize(M, N, acc.data(), row_scale.data(), col_scale, bias ? bias->data() : nullptr,
                   relu, out.data());
      },
      nb::arg("acc").noconvert(), nb::arg("row_scale").noconvert(), nb::arg("col_scale"),
      nb::arg("bias").noconvert().none(), nb::arg("relu"), nb::arg("out").noconvert());

  // int4 weight-only
  nb::class_<PackedW4>(m, "PackedW4")
      .def_ro("M", &PackedW4::M)
      .def_ro("K", &PackedW4::K)
      .def_ro("group", &PackedW4::group)
      .def_prop_ro("nbytes", [](const PackedW4& p) {
        return p.q.size() + p.scales.size() * sizeof(float);
      });
  m.def(
      "make_w4",
      [](In<uint8_t, 2> q, In<float, 2> scales, int K, int group) {
        require(group > 0, "group must be positive");
        const int M = dim(q.shape(0));
        require(dim(q.shape(1)) == (K + 1) / 2, "q must have shape [M, ceil(K/2)]");
        require(dim(scales.shape(0)) == M && dim(scales.shape(1)) == (K + group - 1) / group,
                "scales must have shape [M, ceil(K/group)]");
        PackedW4 p;
        p.M = M;
        p.K = K;
        p.group = group;
        p.q.assign(q.data(), q.data() + q.size());
        p.scales.assign(scales.data(), scales.data() + scales.size());
        return p;
      },
      nb::arg("q").noconvert(), nb::arg("scales").noconvert(), nb::arg("K"), nb::arg("group"));
  m.def(
      "gemm_w4",
      [](const std::string& isa, const PackedW4& a, In<float, 2> b,
         std::optional<In<float, 1>> bias, bool relu, Out<float, 2> out) {
        require(dim(b.shape(0)) == a.K, "b must have shape [K, N]");
        const int N = dim(b.shape(1));
        check_out(out.shape(0), out.shape(1), a.M, N);
        gemm_w4(parse_isa(isa), a, N, b.data(), opt_bias(bias, a), relu, out.data());
      },
      nb::arg("isa"), nb::arg("a"), nb::arg("b").noconvert(), nb::arg("bias").noconvert().none(),
      nb::arg("relu"), nb::arg("out").noconvert());

  // 2:4 structured sparse
  nb::class_<Sparse24>(m, "Sparse24")
      .def_ro("M", &Sparse24::M)
      .def_ro("K", &Sparse24::K)
      .def_prop_ro("nbytes", [](const Sparse24& p) {
        return p.values.size() * sizeof(float) + p.meta.size();
      });
  m.def(
      "make_sparse24",
      [](In<float, 2> values, In<uint8_t, 2> meta) {
        const int M = dim(values.shape(0)), K = 2 * dim(values.shape(1));
        require(dim(meta.shape(0)) == M && dim(meta.shape(1)) == K / 4,
                "meta must have shape [M, K/4] where values has shape [M, K/2]");
        require(K % 4 == 0, "K must be a multiple of 4");
        for (size_t i = 0; i < meta.size(); ++i) {
          const uint8_t mb = meta.data()[i];
          require(mb < 16 && (mb & 3) < ((mb >> 2) & 3), "meta: need idx0 < idx1, both in [0, 3]");
        }
        Sparse24 p;
        p.M = M;
        p.K = K;
        p.values.assign(values.data(), values.data() + values.size());
        p.meta.assign(meta.data(), meta.data() + meta.size());
        return p;
      },
      nb::arg("values").noconvert(), nb::arg("meta").noconvert());
  m.def(
      "gemm_sparse24",
      [](const std::string& isa, const Sparse24& a, In<float, 2> b,
         std::optional<In<float, 1>> bias, bool relu, Out<float, 2> out) {
        require(dim(b.shape(0)) == a.K, "b must have shape [K, N]");
        const int N = dim(b.shape(1));
        check_out(out.shape(0), out.shape(1), a.M, N);
        gemm_sparse24(parse_isa(isa), a, N, b.data(), opt_bias(bias, a), relu, out.data());
      },
      nb::arg("isa"), nb::arg("a"), nb::arg("b").noconvert(), nb::arg("bias").noconvert().none(),
      nb::arg("relu"), nb::arg("out").noconvert());

  // CSR
  nb::class_<Csr>(m, "Csr")
      .def_ro("M", &Csr::M)
      .def_ro("K", &Csr::K)
      .def_prop_ro("nnz", [](const Csr& p) { return p.values.size(); })
      .def_prop_ro("nbytes", [](const Csr& p) {
        return p.values.size() * (sizeof(float) + sizeof(int32_t)) +
               p.row_ptr.size() * sizeof(int32_t);
      });
  m.def(
      "make_csr",
      [](In<float, 1> values, In<int32_t, 1> col, In<int32_t, 1> row_ptr, int K) {
        const int M = dim(row_ptr.shape(0)) - 1;
        require(M >= 0, "row_ptr must have M + 1 entries");
        require(values.shape(0) == col.shape(0), "values and col must have the same length");
        const int32_t* rp = row_ptr.data();
        require(rp[0] == 0 && rp[M] == dim(values.shape(0)), "row_ptr must start at 0, end at nnz");
        for (int i = 0; i < M; ++i) require(rp[i] <= rp[i + 1], "row_ptr must be non-decreasing");
        for (size_t i = 0; i < col.shape(0); ++i)
          require(col.data()[i] >= 0 && col.data()[i] < K, "col index out of range");
        Csr p;
        p.M = M;
        p.K = K;
        p.values.assign(values.data(), values.data() + values.size());
        p.col.assign(col.data(), col.data() + col.size());
        p.row_ptr.assign(rp, rp + row_ptr.size());
        return p;
      },
      nb::arg("values").noconvert(), nb::arg("col").noconvert(), nb::arg("row_ptr").noconvert(),
      nb::arg("K"));
  m.def(
      "gemm_csr",
      [](const std::string& isa, const Csr& a, In<float, 2> b, std::optional<In<float, 1>> bias,
         bool relu, Out<float, 2> out) {
        require(dim(b.shape(0)) == a.K, "b must have shape [K, N]");
        const int N = dim(b.shape(1));
        check_out(out.shape(0), out.shape(1), a.M, N);
        gemm_csr(parse_isa(isa), a, N, b.data(), opt_bias(bias, a), relu, out.data());
      },
      nb::arg("isa"), nb::arg("a"), nb::arg("b").noconvert(), nb::arg("bias").noconvert().none(),
      nb::arg("relu"), nb::arg("out").noconvert());

  // im2col / quantize
  auto im2col_binding = [](auto tag) {
    using T = decltype(tag);
    return [](In<T, 3> x, int kh, int kw, int sh, int sw, int ph, int pw, Out<T, 2> out) {
      const int C = dim(x.shape(0)), H = dim(x.shape(1)), W = dim(x.shape(2));
      require(kh > 0 && kw > 0 && sh > 0 && sw > 0 && ph >= 0 && pw >= 0, "bad conv geometry");
      const int Ho = conv_out_size(H, kh, sh, ph), Wo = conv_out_size(W, kw, sw, pw);
      require(Ho > 0 && Wo > 0, "kernel larger than padded input");
      require(dim(out.shape(0)) == C * kh * kw && dim(out.shape(1)) == Ho * Wo,
              "out must have shape [C*kh*kw, Ho*Wo]");
      im2col<T>(x.data(), C, H, W, kh, kw, sh, sw, ph, pw, out.data());
    };
  };
  m.def("im2col_f32", im2col_binding(float{}), nb::arg("x").noconvert(), nb::arg("kh"),
        nb::arg("kw"), nb::arg("sh"), nb::arg("sw"), nb::arg("ph"), nb::arg("pw"),
        nb::arg("out").noconvert());
  m.def("im2col_s8", im2col_binding(int8_t{}), nb::arg("x").noconvert(), nb::arg("kh"),
        nb::arg("kw"), nb::arg("sh"), nb::arg("sw"), nb::arg("ph"), nb::arg("pw"),
        nb::arg("out").noconvert());
  m.def(
      "quantize_s8",
      [](nb::ndarray<const float, nb::c_contig, nb::device::cpu> x, float scale,
         nb::ndarray<int8_t, nb::c_contig, nb::device::cpu> out) {
        require(scale > 0, "scale must be positive");
        require(x.size() == out.size(), "out must have the same size as x");
        quantize_s8(x.data(), x.size(), 1.0f / scale, out.data());
      },
      nb::arg("x").noconvert(), nb::arg("scale"), nb::arg("out").noconvert());
  m.def(
      "absmax",
      [](nb::ndarray<const float, nb::c_contig, nb::device::cpu> x) {
        return absmax(x.data(), x.size());
      },
      nb::arg("x").noconvert());

  // microbenchmarks
  m.def(
      "peak_fma_gflops",
      [](const std::string& isa, double seconds) { return peak_fma_gflops(parse_isa(isa), seconds); },
      nb::arg("isa"), nb::arg("seconds"));
  m.def(
      "peak_s8_gops",
      [](const std::string& isa, double seconds) { return peak_s8_gops(parse_isa(isa), seconds); },
      nb::arg("isa"), nb::arg("seconds"));
  m.def("triad_gbps", &triad_gbps, nb::arg("bytes_per_array"), nb::arg("reps"));
}
