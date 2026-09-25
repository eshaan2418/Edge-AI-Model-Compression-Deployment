#include <stdexcept>

#include "edgeai/kernels.h"

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace edgeai {

const char* isa_name(Isa isa) {
  switch (isa) {
    case Isa::Scalar:
      return "scalar";
    case Isa::Neon:
      return "neon";
    case Isa::Avx2:
      return "avx2";
    case Isa::Avx512:
      return "avx512";
  }
  return "?";
}

bool isa_supported(Isa isa) {
  switch (isa) {
    case Isa::Scalar:
      return true;
    case Isa::Neon:
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#if defined(__linux__)
      return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
      return true;  // every Apple Silicon core has dotprod
#endif
#else
      return false;
#endif
    case Isa::Avx2:
#if defined(__x86_64__)
      return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
      return false;
#endif
    case Isa::Avx512:
#if defined(__x86_64__)
      return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
      return false;
#endif
  }
  return false;
}

std::vector<Isa> supported_isas() {
  std::vector<Isa> out;
  for (Isa isa : {Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512}) {
    if (isa_supported(isa)) out.push_back(isa);
  }
  return out;
}

Isa best_isa() {
  for (Isa isa : {Isa::Avx512, Isa::Avx2, Isa::Neon}) {
    if (isa_supported(isa)) return isa;
  }
  return Isa::Scalar;
}

Isa parse_isa(const std::string& name) {
  Isa isa;
  if (name == "auto") return best_isa();
  if (name == "scalar") {
    isa = Isa::Scalar;
  } else if (name == "neon") {
    isa = Isa::Neon;
  } else if (name == "avx2") {
    isa = Isa::Avx2;
  } else if (name == "avx512") {
    isa = Isa::Avx512;
  } else {
    throw std::invalid_argument("unknown isa '" + name + "'");
  }
  if (!isa_supported(isa)) {
    throw std::invalid_argument(std::string("isa '") + name + "' not supported on this CPU");
  }
  return isa;
}

}  // namespace edgeai
