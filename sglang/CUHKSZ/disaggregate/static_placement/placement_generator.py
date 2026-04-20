from collections import defaultdict
import torch
import numpy as np

# ---------- Helper functions for different placement policies ----------

def _init_route_experts(num_gpus: int, num_expert_per_gpu: int, num_logical_experts: int):
    """Average partition of logical experts to GPUs as route_experts."""
    gpu_configs = []
    experts_per_gpu_base = num_logical_experts // num_gpus
    extra = num_logical_experts % num_gpus
    logical_idx = 0

    for gpu in range(num_gpus):
        count = experts_per_gpu_base + (1 if gpu < extra else 0)
        route_experts = list(range(logical_idx, logical_idx + count))
        logical_idx += count
        gpu_configs.append({"route_experts": route_experts, "redundant_experts": []})
    return gpu_configs


def _assign_redundancy_ring(
    gpu_configs,
    num_gpus: int,
    num_expert_per_gpu: int,
    redundant: bool = True,
):
    """Policy 1: simple ring-based redundancy, ignoring statistics."""
    if redundant == False:
        # Explicitly skip adding redundant experts when requested.
        for gpu in range(num_gpus):
            gpu_configs[gpu]["redundant_experts"] = []
        return

    for gpu in range(num_gpus):
        current_set = set(gpu_configs[gpu]["route_experts"])
        total_needed = num_expert_per_gpu - len(current_set)
        redundant_list = []

        offset = 1
        while len(redundant_list) < total_needed:
            src_gpu = (gpu + offset) % num_gpus
            for e in gpu_configs[src_gpu]["route_experts"]:
                if e not in current_set:
                    redundant_list.append(e)
                    current_set.add(e)
                    if len(redundant_list) >= total_needed:
                        break
            offset += 1
            if offset > num_gpus:
                break

        gpu_configs[gpu]["redundant_experts"] = redundant_list


def _compute_algorithm1_replicas(
    num_gpus: int,
    num_expert_per_gpu: int,
    num_logical_experts: int,
    expert_counts: dict,
):
    """Run Algorithm 1 to decide how many replicas each expert should have."""
    S = num_gpus * num_expert_per_gpu  # Total slots
    R = S - num_logical_experts  # Remaining redundancy slots

    # Initialize: each expert gets at least 1 replica
    replicas = {i: 1 for i in range(num_logical_experts)}

    # Ensure counts cover all experts
    counts = {i: expert_counts.get(i, 0.0) for i in range(num_logical_experts)}

    # Greedy assignment based on unit_load
    while R > 0:
        # 约束：任何一个 expert 的副本数不能超过 num_gpus（最多放在每个 GPU 上各一份）
        eligible_experts = [i for i in range(num_logical_experts) if replicas[i] < num_gpus]
        if not eligible_experts:
            # 理论上不会发生（否则总 S 太大），但为了健壮性直接退出循环
            break

        unit_loads = {}
        for i in eligible_experts:
            if replicas[i] > 0:
                unit_loads[i] = counts[i] / replicas[i]
            else:
                unit_loads[i] = float("inf") if counts[i] > 0 else 0.0

        e = max(unit_loads.keys(), key=lambda x: unit_loads[x])
        replicas[e] += 1
        R -= 1

    return replicas, counts


def _assign_redundancy_unit_load(
    gpu_configs,
    num_gpus: int,
    num_expert_per_gpu: int,
    num_logical_experts: int,
    replicas: dict,
    counts: dict,
):
    """Policy 2: place redundancy using Algorithm-1 replica counts, unit-load based."""
    expert_remaining_replicas = {i: replicas[i] - 1 for i in range(num_logical_experts)}
    expert_allocated_count = {i: 1 for i in range(num_logical_experts)}

    def get_unit_load(expert_id: int) -> float:
        allocated = expert_allocated_count[expert_id]
        if allocated > 0:
            return counts[expert_id] / allocated
        return float("inf") if counts[expert_id] > 0 else 0.0

    for gpu in range(num_gpus):
        current_set = set(gpu_configs[gpu]["route_experts"])
        total_needed = num_expert_per_gpu - len(current_set)
        redundant = []

        while len(redundant) < total_needed:
            candidates = []
            for expert_id in range(num_logical_experts):
                if expert_remaining_replicas[expert_id] > 0 and expert_id not in current_set:
                    candidates.append((expert_id, get_unit_load(expert_id)))

            if not candidates:
                # All algorithm-1 replicas placed, fill remaining with any experts
                for expert_id in range(num_logical_experts):
                    if len(redundant) >= total_needed:
                        break
                    if expert_id not in current_set:
                        redundant.append(expert_id)
                        current_set.add(expert_id)
                break

            best_expert = max(candidates, key=lambda x: x[1])[0]
            redundant.append(best_expert)
            current_set.add(best_expert)
            expert_remaining_replicas[best_expert] -= 1
            expert_allocated_count[best_expert] += 1

        gpu_configs[gpu]["redundant_experts"] = redundant


def _assign_redundancy_co_activation(
    gpu_configs,
    num_gpus: int,
    num_expert_per_gpu: int,
    num_logical_experts: int,
    replicas: dict,
    counts: dict,
    co_occurrence_counts: dict,
):
    """Policy 3: co-activation-aware placement on top of Algorithm-1 replica counts."""
    expert_remaining_replicas = {i: replicas[i] - 1 for i in range(num_logical_experts)}
    expert_allocated_count = {i: 1 for i in range(num_logical_experts)}

    gpu_assignment_sets = [set(cfg["route_experts"]) for cfg in gpu_configs]
    gpu_redundant_lists = [[] for _ in range(num_gpus)]
    gpu_slots_remaining = [num_expert_per_gpu - len(cfg["route_experts"]) for cfg in gpu_configs]

    def unit_load_for_order(expert_id: int) -> float:
        rep = replicas[expert_id]
        return counts[expert_id] / rep if rep > 0 else 0.0

    expert_order = sorted(range(num_logical_experts), key=unit_load_for_order, reverse=True)

    def incremental_co_cost(expert_id: int, gpu_id: int) -> float:
        cost = 0.0
        co_map = co_occurrence_counts.get(expert_id, {})
        for other in gpu_assignment_sets[gpu_id]:
            cost += co_map.get(other, 0.0)
        return cost

    for expert_id in expert_order:
        copies_needed = expert_remaining_replicas[expert_id]
        for _ in range(copies_needed):
            best_gpu = None
            best_cost = None
            for gpu in range(num_gpus):
                if gpu_slots_remaining[gpu] <= 0:
                    continue
                if expert_id in gpu_assignment_sets[gpu]:
                    continue
                cost = incremental_co_cost(expert_id, gpu)
                if (
                    best_gpu is None
                    or cost < best_cost
                    or (cost == best_cost and gpu_slots_remaining[gpu] > gpu_slots_remaining[best_gpu])
                ):
                    best_gpu = gpu
                    best_cost = cost

            if best_gpu is None:
                # fallback：通过“换位”来为该 expert 腾出一个 GPU 槽位
                swap_choice = None
                swap_cost_best = None

                # 我们希望：在某个 GPU g 上放入 expert_id，同时把 g 上一个冗余 expert o
                # 挪到另一个有空位的 GPU h 上，且不在同一个 GPU 上重复 expert。
                for g in range(num_gpus):
                    # 目标 GPU g 不能已经有这个 expert
                    if expert_id in gpu_assignment_sets[g]:
                        continue
                    # 只能在已满的 GPU 上做腾挪（否则前面就能直接 best_gpu = g 了）
                    if gpu_slots_remaining[g] != 0:
                        continue

                    # 只在冗余 experts 上做交换，不动 route_experts
                    for o in list(gpu_redundant_lists[g]):
                        # 为 o 找一个新的 GPU h
                        best_h = None
                        best_h_cost = None
                        for h in range(num_gpus):
                            if h == g:
                                continue
                            if gpu_slots_remaining[h] <= 0:
                                continue
                            if o in gpu_assignment_sets[h]:
                                continue
                            cost_o_h = incremental_co_cost(o, h)
                            if best_h is None or cost_o_h < best_h_cost:
                                best_h = h
                                best_h_cost = cost_o_h

                        if best_h is None:
                            continue

                        # 近似评估该 swap 的代价：E 放在 g + o 放在 h 的新增 co-activation
                        cost_e_g = incremental_co_cost(expert_id, g)
                        total_swap_cost = cost_e_g + best_h_cost

                        if swap_choice is None or total_swap_cost < swap_cost_best:
                            swap_choice = (g, o, best_h)
                            swap_cost_best = total_swap_cost

                if swap_choice is not None:
                    g, o, h = swap_choice
                    # 从 g 移除 o
                    gpu_redundant_lists[g].remove(o)
                    gpu_assignment_sets[g].remove(o)
                    gpu_slots_remaining[g] += 1  # g 现在有一个空位

                    # 在 g 放入 expert_id
                    gpu_redundant_lists[g].append(expert_id)
                    gpu_assignment_sets[g].add(expert_id)
                    gpu_slots_remaining[g] -= 1
                    expert_allocated_count[expert_id] += 1

                    # 把 o 放到新的 GPU h
                    gpu_redundant_lists[h].append(o)
                    gpu_assignment_sets[h].add(o)
                    gpu_slots_remaining[h] -= 1

                    # 这一轮 expert 的一个副本已经放好，继续下一轮副本
                    continue
                else:
                    # 理论上在 replicas[i] <= num_gpus 的前提下不应该发生；
                    # 兜底：仍然抛错，方便排查。
                    raise RuntimeError(
                        f"Unable to assign redundant expert {expert_id}: "
                        "no feasible swap to avoid duplicating on the same GPU."
                    )

            gpu_redundant_lists[best_gpu].append(expert_id)
            gpu_assignment_sets[best_gpu].add(expert_id)
            gpu_slots_remaining[best_gpu] -= 1
            expert_allocated_count[expert_id] += 1

    for gpu in range(num_gpus):
        gpu_configs[gpu]["redundant_experts"] = gpu_redundant_lists[gpu]


# ---------- Helper functions for printing / analysis ----------

def _print_gpu_mapping(gpu_configs):
    """Pretty-print GPU -> experts mapping."""
    print("GPU to expert mapping:")
    BLUE = "\033[94m"
    RESET = "\033[0m"

    for gpu_id, cfg in enumerate(gpu_configs):
        total = len(cfg["route_experts"]) + len(cfg["redundant_experts"])
        string1 = f"EP{gpu_id}:{total} = {len(cfg['route_experts'])}+{len(cfg['redundant_experts'])}R  "
        string2 = f"{cfg['route_experts']}"
        string3 = f"{BLUE}{cfg['redundant_experts']}{RESET}"
        print(f"{string1}{string2} + {string3}")


def _print_replication_and_stats(
    gpu_configs,
    expert_to_gpus,
    num_logical_experts: int,
    expert_counts: dict | None,
    co_occurrence_counts: dict | None,
):
    """Print replication histogram, GPU co-activation stats, and top experts."""
    from collections import Counter
    import statistics

    # Replication summary
    print("\nExpert Replication Summary:")
    replica_count = Counter(len(v) for v in expert_to_gpus.values())
    total = sum(replica_count.values())
    for k in sorted(replica_count.keys()):
        expert_list = [e for e in expert_to_gpus.keys() if len(expert_to_gpus[e]) == k]
        if k == 1:
            print(f"Experts appear {k} time: {replica_count[k]} experts ({replica_count[k] / total * 100:.1f}%)")
        else:
            print(
                f"Experts appear {k} times: {replica_count[k]} experts "
                f"({replica_count[k] / total * 100:.1f}%), {expert_list}"
            )

    # GPU co-activation stats (for any policy, if data is provided)
    if co_occurrence_counts:
        print("\nGPU Co-activation Counts:")
        gpu_totals = compute_gpu_co_activation_totals(gpu_configs, co_occurrence_counts)
        for gpu_id, s in enumerate(gpu_totals):
            print(f"GPU{gpu_id}: Σ = {int(s)}")
        if gpu_totals:
            std = statistics.stdev(gpu_totals) if len(gpu_totals) > 1 else 0.0
            print(
                f"Max: {int(max(gpu_totals))}, Min: {int(min(gpu_totals))}, "
                f"Avg: {int(sum(gpu_totals) / len(gpu_totals))}, STD: {int(std)}"
            )

    # Top experts by replicas / load (only if we have activation counts)
    if expert_counts is not None:
        total_activation = sum(expert_counts.values())
        if total_activation <= 0:
            return

        expert_replica_counts = {i: len(expert_to_gpus[i]) for i in range(num_logical_experts)}
        expert_info = []
        for expert_id in range(num_logical_experts):
            replica_num = expert_replica_counts[expert_id]
            activation_count = expert_counts.get(expert_id, 0.0)
            activation_ratio = (activation_count / total_activation * 100) if total_activation > 0 else 0.0
            unit_load = activation_count / replica_num if replica_num > 0 else 0.0
            expert_info.append(
                {
                    "expert_id": expert_id,
                    "replicas": replica_num,
                    "activation_count": activation_count,
                    "activation_ratio": activation_ratio,
                    "unit_load": unit_load,
                }
            )

        expert_info.sort(key=lambda x: (-x["replicas"], -x["activation_count"]))
        high_replica_experts = [e for e in expert_info if e["replicas"] > 1]
        if high_replica_experts:
            print("\nTop 10 Most Replicated Experts:")
            for i, e in enumerate(high_replica_experts[:10], 1):
                print(
                    f"  {i:2d}. Expert {e['expert_id']:3d}: {e['replicas']} replicas, "
                    f"load={int(e['activation_count'])} ({e['activation_ratio']:.2f}%), "
                    f"unit_load={int(e['unit_load'])}"
                )


# ---------- Main API ----------
def generate_expert_mapping(
    num_gpus: int,
    num_expert_per_gpu: int,
    num_logical_experts: int,
    expert_counts: dict = None,
    co_occurrence_counts: dict = None,
    use_co_activation_placement: bool = False,
    print_detail: bool = False,
    redundant: bool = True,
):
    # num_gpus = self.ep_gpu_used,
    # num_expert_per_gpu = self.num_expert_per_gpu,
    # num_logical_experts = self.num_logical_experts,
    # expert_counts = None,
    # co_occurrence_counts = None,
    # print_detail = True,
    # redundant = True,
    """
    Generate expert mapping with redundancy assignment.

    Args:
        num_gpus: Number of GPUs
        num_expert_per_gpu: Number of experts per GPU
        num_logical_experts: Number of logical experts
        redundant: Controls redundant placement under ring policy. Pass None to
                    skip adding redundant experts.
        expert_counts: Dict mapping expert_id to activation count (for Algorithm 1)
                       If None, uses circular distribution (old strategy)
        co_occurrence_counts: Nested dict capturing co-activation counts between experts.
                              Provide to report co-activation totals and/or enable placement.
        use_co_activation_placement: If True (and co_occurrence_counts provided), use
                                     co-activation-aware placement for redundancy.
        print_detail: Whether to print detailed information
    """

    expert_to_gpus = defaultdict(set)
    expert_gpu_to_phys = defaultdict(dict)

    # route_experts
    gpu_configs = _init_route_experts(num_gpus, num_expert_per_gpu, num_logical_experts)

    # redundancy according to policy
    if expert_counts is None:
        # Policy 1 🌈: ring-based
        _assign_redundancy_ring(gpu_configs, num_gpus, num_expert_per_gpu, redundant=redundant)
    else:
        # Policy 2 / 3: Algorithm 1 replicas
        replicas, counts = _compute_algorithm1_replicas(
            num_gpus, num_expert_per_gpu, num_logical_experts, expert_counts
        )

        if use_co_activation_placement and co_occurrence_counts:
            _assign_redundancy_co_activation(
                gpu_configs,
                num_gpus,
                num_expert_per_gpu,
                num_logical_experts,
                replicas,
                counts,
                co_occurrence_counts,
            )
        else:
            _assign_redundancy_unit_load(
                gpu_configs,
                num_gpus,
                num_expert_per_gpu,
                num_logical_experts,
                replicas,
                counts,
            )

    # 建立 expert_to_gpus / expert_gpu_to_phys
    phys_counter = 0
    for gpu_id, cfg in enumerate(gpu_configs):
        for logical_id in cfg["route_experts"] + cfg["redundant_experts"]:
            expert_to_gpus[logical_id].add(gpu_id)
            expert_gpu_to_phys[logical_id][gpu_id] = phys_counter
            phys_counter += 1
    
    if print_detail:
        _print_gpu_mapping(gpu_configs)
        _print_replication_and_stats(
            gpu_configs,
            expert_to_gpus,
            num_logical_experts,
            expert_counts,
            co_occurrence_counts,
        )

    for gpu_id in range(num_gpus):
        expert_set = set()
        for logical_id in gpu_configs[gpu_id]["route_experts"] + gpu_configs[gpu_id]["redundant_experts"]:
            if logical_id in expert_set:
                raise ValueError(f"Expert {logical_id} appears multiple times on GPU {gpu_id}")
            expert_set.add(logical_id)
            
    return expert_to_gpus, expert_gpu_to_phys


def load_expert_counts_from_json(json_path: str):
    """
    Load expert activation counts from JSON statistics file.
    
    Args:
        json_path: Path to JSON file containing expert_frequencies
        
    Returns:
        Dict mapping expert_id to activation count
    """
    import json
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    expert_counts = {}
    if "expert_frequencies" in data:
        for item in data["expert_frequencies"]:
            expert_counts[item["expert_id"]] = item["frequency"]
    
    return expert_counts


def load_co_occurrence_from_json(json_path: str):
    """
    Load expert co-activation (pair) statistics from JSON file.

    Returns:
        Dict[expert_i][expert_j] = co-activation count
    """
    import json
    from collections import defaultdict

    with open(json_path, 'r') as f:
        data = json.load(f)

    co_occurrence = defaultdict(dict)
    for item in data.get("expert_pairs", []):
        i = item["expert_i"]
        j = item["expert_j"]
        count = item["count"]
        co_occurrence[i][j] = co_occurrence[i].get(j, 0.0) + count
        co_occurrence[j][i] = co_occurrence[j].get(i, 0.0) + count

    return co_occurrence


def compute_gpu_co_activation_totals(gpu_configs, co_occurrence_counts):
    """
    For each GPU, compute the total co-activation score among all assigned experts
    (route + redundant). co_occurrence_counts should be a symmetric dict.
    """
    gpu_totals = []
    for cfg in gpu_configs:
        experts = cfg["route_experts"] + cfg["redundant_experts"]
        total = 0.0
        for idx, expert_id in enumerate(experts):
            co_map = co_occurrence_counts.get(expert_id, {})
            for other in experts[idx + 1:]:
                total += co_map.get(other, 0.0)
        gpu_totals.append(total)
    return gpu_totals