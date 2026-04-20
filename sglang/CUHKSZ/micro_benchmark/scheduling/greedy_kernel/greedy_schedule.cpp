#include <torch/extension.h>

// 声明 CUDA 函数
torch::Tensor called_experts_cuda(torch::Tensor topk_id, int num_experts);

torch::Tensor greedy_schedule_cuda(
    torch::Tensor topk_id,
    torch::Tensor called_ids,
    torch::Tensor expert2gpus,
    torch::Tensor expert2phys,
    torch::Tensor copy_count,
    int num_experts,
    int num_gpus,
    bool all_expert_have_replica
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("called_experts", &called_experts_cuda, "Called Experts (CUDA)");
    m.def("greedy_schedule", &greedy_schedule_cuda, "Greedy Schedule (CUDA)");
}
