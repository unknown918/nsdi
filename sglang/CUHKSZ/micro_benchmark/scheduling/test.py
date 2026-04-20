import torch
import called_experts


topk_id = torch.tensor([11, 23, 33, 81, 99, 35, 113, 115, 135, 136, 142, 102], dtype=torch.int32, device="cuda")
num_experts = 160

called_experts_ids = called_experts.called_experts(topk_id, num_experts)
print(called_experts_ids)