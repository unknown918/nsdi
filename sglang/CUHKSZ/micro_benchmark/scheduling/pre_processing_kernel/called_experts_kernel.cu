#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mark_called_experts_kernel(const int32_t* __restrict__ topk_id,
                                           bool* __restrict__ mask,
                                           int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int expert_id = topk_id[idx];
        mask[expert_id] = true;  // 幂等写，不需要原子操作
    }
}

torch::Tensor called_experts_cuda(torch::Tensor topk_id, int num_experts) {
    TORCH_CHECK(topk_id.is_cuda(), "topk_id must be CUDA tensor");
    TORCH_CHECK(topk_id.dtype() == torch::kInt32, "topk_id must be int32");

    auto mask = torch::zeros({num_experts}, torch::dtype(torch::kBool).device(topk_id.device()));

    int total = topk_id.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    mark_called_experts_kernel<<<blocks, threads>>>(
        topk_id.data_ptr<int32_t>(),
        mask.data_ptr<bool>(),
        total
    );

    auto ids = torch::nonzero(mask).squeeze(1);

    // return mask;
    return ids;
}
