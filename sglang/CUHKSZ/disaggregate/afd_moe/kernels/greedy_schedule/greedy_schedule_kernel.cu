#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------
// Kernel 1: 标记被调用的 experts
// ---------------------------------------------------
__global__ void mark_called_experts_kernel(const int32_t* __restrict__ topk_id,
                                           bool* __restrict__ mask,
                                           int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int expert_id = topk_id[idx];
        mask[expert_id] = true;  // 幂等写，不需要原子操作
    }
}

// ---------------------------------------------------
// Kernel 2: 贪心调度 (单线程执行)
// ---------------------------------------------------
// Kernel 2: 贪心调度 (单线程执行，两阶段)
__global__ void greedy_schedule_kernel(const int32_t* __restrict__ called_ids,
                                       int num_called,
                                       const int32_t* __restrict__ expert2gpus,
                                       const int32_t* __restrict__ expert2phys,
                                       const int32_t* __restrict__ copy_count, 
                                       int max_copies,
                                       int32_t* __restrict__ logical2phys,
                                       const int32_t* __restrict__ topk_id,
                                       int total_topk,
                                       int num_gpus,
                                       bool all_expert_have_replica) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int load[16];  // 假设 <= 256 GPUs
        for (int i = 0; i < num_gpus; i++) load[i] = 0;

        // --------------------------
        // Step 1: 处理无冗余 experts
        // --------------------------
        // Step 1: 无冗余专家
        if (!all_expert_have_replica) {
            for (int idx = 0; idx < num_called; idx++) {
                int lid = called_ids[idx];
                if (copy_count[lid] == 1) {
                    int g = expert2gpus[lid * max_copies + 0];
                    int p = expert2phys[lid * max_copies + 0];
                    logical2phys[lid] = p;
                    load[g] += 1;
                }
            }
        }
        // --------------------------
        // Step 2: 处理有冗余 experts
        // --------------------------
        for (int idx = 0; idx < num_called; idx++) {
            int lid = called_ids[idx];
            if (logical2phys[lid] != -1) continue;  // 已经分配过

            int chosen_gpu = -1;
            int chosen_phys = -1;
            int best_load = 1e9;

            // 直接在一轮循环里完成候选枚举和最优选择
            for (int c = 0; c < max_copies; c++) {
                int g = expert2gpus[lid * max_copies + c];
                int p = expert2phys[lid * max_copies + c];
                if (g >= 0 && p >= 0) {
                    int l = load[g];
                    if (l < best_load) {
                        best_load = l;
                        chosen_gpu = g;
                        chosen_phys = p;
                    }
                }
            }

            logical2phys[lid] = chosen_phys;
            load[chosen_gpu] += 1;
        }

    }
}

// ---------------------------------------------------
// Kernel 3: 映射 topk_id
// ---------------------------------------------------

__global__ void map_topk_kernel(const int32_t* __restrict__ topk_id,
                                const int32_t* __restrict__ logical2phys,
                                int32_t* __restrict__ mapped_topk,
                                int total_topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_topk) {
        int lid = topk_id[idx];
        mapped_topk[idx] = logical2phys[lid];
    }
}


// ---------------------------------------------------
// Wrapper: called_experts
// ---------------------------------------------------
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

    auto ids = torch::nonzero(mask).squeeze(1).to(torch::kInt32);
    return ids;
}

// ---------------------------------------------------
// Wrapper: greedy_schedule
// ---------------------------------------------------
torch::Tensor greedy_schedule_cuda(
    torch::Tensor topk_id,
    torch::Tensor called_ids,
    torch::Tensor expert2gpus,
    torch::Tensor expert2phys,
    torch::Tensor copy_count,
    int num_experts,
    int num_gpus,
    bool all_expert_have_replica
) {
    TORCH_CHECK(topk_id.is_cuda(), "topk_id must be CUDA tensor");
    TORCH_CHECK(called_ids.is_cuda(), "called_ids must be CUDA tensor");

    int max_copies = expert2gpus.size(1);
    int total_topk = topk_id.numel();

    auto logical2phys = torch::full({num_experts}, -1, torch::dtype(torch::kInt32).device(topk_id.device()));
    auto mapped_topk = torch::full(
            topk_id.sizes(), -1,
            torch::dtype(torch::kInt32).device(topk_id.device())
        );
    greedy_schedule_kernel<<<1, 1>>>(
        called_ids.data_ptr<int32_t>(),
        called_ids.size(0),
        expert2gpus.data_ptr<int32_t>(),
        expert2phys.data_ptr<int32_t>(),
        copy_count.data_ptr<int32_t>(),
        max_copies,
        logical2phys.data_ptr<int32_t>(),
        topk_id.data_ptr<int32_t>(),
        total_topk,
        num_gpus,
        all_expert_have_replica
    );

    int threads = 256;
    int blocks = (total_topk + threads - 1) / threads;
    map_topk_kernel<<<blocks, threads>>>(
        topk_id.data_ptr<int32_t>(),
        logical2phys.data_ptr<int32_t>(),
        mapped_topk.data_ptr<int32_t>(),
        total_topk
    );

    return mapped_topk;
}
