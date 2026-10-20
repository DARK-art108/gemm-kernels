#include <torch/extension.h>

torch::Tensor fused_rmsnorm_swiglu_cuda(torch::Tensor input,
                                        torch::Tensor weight,
                                        torch::Tensor gate);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_rmsnorm_swiglu", &fused_rmsnorm_swiglu_cuda,
        "Fused RMSNorm + SwiGLU (CUDA)");
}