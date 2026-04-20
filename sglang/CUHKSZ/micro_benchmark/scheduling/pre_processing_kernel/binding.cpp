#include <torch/extension.h>

// 声明 CUDA 实现
torch::Tensor called_experts_cuda(torch::Tensor topk_id, int num_experts);

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("called_experts", &called_experts_cuda, "Mark called experts (CUDA)");
}
