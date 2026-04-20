import torch

def count_experts_per_gpu(topk_id, num_experts, num_gpus):
    """
    统计每个 GPU 上被调用的 expert 数量

    Args:
        topk_id (torch.Tensor): shape [batch, topk]，存放 expert id
        num_experts (int): 总 expert 数量
        num_gpus (int): GPU 数量

    Returns:
        torch.Tensor: [num_gpus]，每个 GPU 上的 expert 使用次数
    """
    # 展平，取唯一值（去重后统计哪些 expert 被调用）
    unique_experts = torch.unique(topk_id)

    # 计算每个 expert 属于哪个 GPU
    experts_per_gpu = num_experts // num_gpus
    expert_to_gpu = unique_experts // experts_per_gpu

    # 避免整除不均，clip 在 [0, num_gpus-1]
    expert_to_gpu = torch.clamp(expert_to_gpu, max=num_gpus - 1)

    # 统计每个 GPU 上的 expert 数量
    counts = torch.bincount(expert_to_gpu, minlength=num_gpus)

    return counts


if __name__ == "__main__":
    # 示例数据
    topk_id = torch.tensor([
[  7,  88,  96, 149, 152,  11],
        [ 23,  35,  37,  67,  73, 141],
        [ 32,  35, 133, 137, 155,  37],
        [ 43,  83,  85,  91, 150,  92],
        [ 26,  42,  43, 106, 112,  56],
        [ 55, 121, 122, 150,  44, 154],
        [ 42,  43,  96, 105, 118,  89],
        [ 52,  85,  91,  92, 138, 121],
        [ 43,  82,  85,  92, 138,  41],
        [  9,  17,  23,  27,  58,  39],
        [ 11,  15,  44,  50, 139,  17],
        [ 22,  64,  72,  74,  82,  95],
        [  5,  17, 139, 149, 157,   4],
        [  2,   7,  96,  99, 137,  86],
        [ 35,  36,  92,  96, 114,  23],
        [  5,  15,  17,  46,  60,  50],
        [ 36,  95, 107, 108, 110,  23],
        [  3,   4,  13,  26,  89,  17],
        [ 43,  54,  97, 106, 112,  59],
        [  0,   2, 123, 124, 144, 146],
        [ 42,  43, 106, 112, 138,  40],
        [ 74,  86,  89, 146, 147,  85],
        [ 29,  31,  54,  93,  96,  98],
        [ 60,  61, 101, 151, 153,  75],
        [  3,   4,  13,  23,  89,  11],
        [ 82, 136, 142, 145, 147, 124],
        [ 24,  38,  67,  75, 115,  72],
        [ 34,  69,  90,  96,  98,  71],
        [  6,  11,  13,  75, 138, 121],
        [ 85,  91, 107, 110, 128,  88],
        [ 17,  44,  46,  55, 109,  16],
        [ 25,  36,  40,  95,  96,  85],
        [ 17,  90,  93, 148, 152, 140]
        ], device="cuda", dtype=torch.int32)

    num_experts = 160
    num_gpus = 8

    result = count_experts_per_gpu(topk_id, num_experts, num_gpus)
    print("每个 GPU 上被调用的 expert 数量:", result.tolist())
