import argparse
import json
import os
import signal
import subprocess
import sys
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GlobalConfig:
    configs: List[str] = field(default_factory=lambda: ["1:6", "1:8", "2:6", "2:8", "4:6", "4:8", "8:6", "8:8"])
    bench_mcs: List[int] = field(default_factory=lambda: [4, 16, 64, 256, 512, 1024, 2048])
    warmup_mc: int = 8
    warmup_rounds: int = 3
    input_len: int = 64
    output_len: int = 16
    wait_secs: int = 300
    bench_timeout: int = 1200


@dataclass
class Experiment:
    name: str
    enable: bool = True
    configs: Optional[List[str]] = None
    bench_mcs: Optional[List[int]] = None


@dataclass
class Runtime:
    rank: int
    world_size: int
    master_addr: str
    master_port: int
    model_path: str
    model_name: str
    data_path: str
    log_root: Path
    experiments_json: Path
    gpus_per_node: int = 8
    python_bin: str = sys.executable


class ProcGroup:
    def __init__(self) -> None:
        self.procs: List[subprocess.Popen] = []

    def add(self, p: subprocess.Popen) -> None:
        self.procs.append(p)

    def stop_all(self, sig: int = signal.SIGTERM, grace: float = 5.0) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.time() + grace
        for p in self.procs:
            if p.poll() is None:
                while time.time() < deadline and p.poll() is None:
                    time.sleep(0.2)
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        self.procs.clear()


def load_experiments(cfg_path: Path, g: GlobalConfig) -> List[Experiment]:
    if not cfg_path.exists():
        return [Experiment(name="default", configs=g.configs, bench_mcs=g.bench_mcs)]
    raw = json.loads(cfg_path.read_text())
    graw = raw.get("global", {})

    for key in [
        "configs", "bench_mcs", "warmup_mc", "warmup_rounds",
        "input_len", "output_len", "wait_secs", "bench_timeout"
    ]:
        if key in graw:
            setattr(g, key, graw[key])

    exps = []
    for i, e in enumerate(raw.get("experiments", [])):
        exps.append(
            Experiment(
                name=str(e.get("name", f"exp_{i}")),
                enable=bool(e.get("enable", True)),
                configs=e.get("configs"),
                bench_mcs=e.get("bench_mcs"),
            )
        )
    return exps


def wait_health(port: int, timeout_s: int, server: subprocess.Popen) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if server.poll() is not None: return False
        try:
            out = subprocess.run(
                ["curl", "-s", f"http://127.0.0.1:{port}/health"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            if out.returncode == 0: return True
        except:
            pass
        time.sleep(2)
    return False


def send_signal(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def wait_signal(path: Path, timeout_s: int, abort_dir: Optional[Path] = None) -> bool:
    end = time.time() + timeout_s
    parent = path.parent
    while time.time() < end:
        try:
            os.listdir(parent)
        except:
            pass
        if path.exists(): return True
        if abort_dir and any(abort_dir.glob("abort_rank*")): return False
        time.sleep(2)
    return False


def server_cmd(rt: Runtime, tp: int, ep: int, role: str, attn_idx: int = 0, expert_idx: int = -1) -> List[str]:
    port = 30010 if role == "attn" else 30020
    attn_node = attn_idx if role == "attn" else -1
    expert_node = expert_idx if role == "expert" else -1

    return [
        rt.python_bin, "-m", "sglang.launch_server",
        "--model-path", rt.model_path,
        "--port", str(port),
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--enable-dp-attention",
        "--enable-disaggregation",
        "--tp-size", str(tp),
        "--ep-size", str(ep),
        "--nnodes", str(rt.world_size),
        "--node-rank", str(rt.rank),
        "--attention-node", str(attn_node),
        "--expert-node", str(expert_node),
        "--dist-init-addr", f"{rt.master_addr}:{rt.master_port}",
        "--sampling-backend", "pytorch",
        "--mem-fraction-static", "0.95",
    ]


def bench_cmd(rt: Runtime, g: GlobalConfig, mc: int, output_jsonl: Path) -> List[str]:
    return [
        rt.python_bin, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--num-prompts", str(mc),
        "--random-input-len", str(g.input_len),
        "--random-output-len", str(g.output_len),
        "--random-range-ratio", "1.0",
        "--dataset-name", "random",
        "--dataset-path", rt.data_path,
        "--max-concurrency", str(mc),
        "--seed", "42", "--host", "127.0.0.1", "--port", "30010",
        "--output-file", str(output_jsonl),
        "--model", rt.model_name,
    ]


def run_bench_monitored(rt: Runtime, g: GlobalConfig, mc: int, output_jsonl: Path,
                        stdout_log: Path, server: subprocess.Popen, fail_signal: Path) -> int:
    cmd = bench_cmd(rt, g, mc, output_jsonl)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    start = time.time()
    while proc.poll() is None:
        if fail_signal.exists() or server.poll() is not None:
            send_signal(fail_signal)
            os.killpg(proc.pid, signal.SIGKILL)
            return 3
        if time.time() - start > g.bench_timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            return 124
        time.sleep(5)
    return proc.returncode or 0


def run_rank0_one_config(rt: Runtime, g: GlobalConfig, exp_dir: Path, signal_dir: Path,
                         cfg_idx: int, tp: int, ep: int, mcs: List[int]) -> None:
    cfg_dir = exp_dir / f"tp{tp}_ep{ep}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    group = ProcGroup()

    attn_log = cfg_dir / "attention_server.log"
    with attn_log.open("w") as f:
        p = subprocess.Popen(server_cmd(rt, tp, ep, "attn", attn_idx=0),
                             stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    group.add(p)

    if not wait_health(30010, g.wait_secs, p):
        print(f"[ERROR] Config {tp}:{ep} server failed to start")
        group.stop_all(sig=signal.SIGKILL)
    else:
        for i in range(1, g.warmup_rounds + 1):
            subprocess.run(bench_cmd(rt, g, g.warmup_mc, cfg_dir / f"warmup_{i}.jsonl"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for mc in mcs:
            log = cfg_dir / f"bench_mc{mc}.log"
            jsonl = cfg_dir / f"bench_mc{mc}.jsonl"
            rc = run_bench_monitored(rt, g, mc, jsonl, log, p, signal_dir / f"config_fail_{cfg_idx}")
            if rc != 0: break

        group.stop_all()

    send_signal(signal_dir / f"done_{cfg_idx}")
    for r in range(1, rt.world_size):
        wait_signal(signal_dir / f"ack_{cfg_idx}_rank{r}", timeout_s=600, abort_dir=signal_dir.parent)


def run_worker_one_config(rt: Runtime, exp_dir: Path, signal_dir: Path, cfg_idx: int, tp: int, ep: int) -> None:
    cfg_dir = exp_dir / f"tp{tp}_ep{ep}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    group = ProcGroup()

    attn_nodes = (tp + rt.gpus_per_node - 1) // rt.gpus_per_node
    role = "attn" if rt.rank < attn_nodes else "expert"
    expert_idx = rt.rank - attn_nodes

    log = cfg_dir / f"{role}_rank{rt.rank}.log"
    with log.open("w") as f:
        p = subprocess.Popen(server_cmd(rt, tp, ep, role, attn_idx=rt.rank, expert_idx=expert_idx),
                             stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    group.add(p)

    wait_signal(signal_dir / f"done_{cfg_idx}", timeout_s=3600, abort_dir=signal_dir.parent)
    group.stop_all(sig=signal.SIGKILL)
    send_signal(signal_dir / f"ack_{cfg_idx}_rank{rt.rank}")


def parse_args() -> Runtime:
    p = argparse.ArgumentParser(description="Distributed SGLang Benchmark")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--log-root", default="./logs_autojob")
    p.add_argument("--experiments-json", default="./experiments.json")
    ns = p.parse_args()

    rank = os.environ.get("RANK")
    if not rank:
        raise RuntimeError("RANK not set by platform")
    rank = int(rank)

    world_size = os.environ.get("WORLD_SIZE")
    if not world_size:
        raise RuntimeError("WORLD_SIZE not set by platform")
    world_size = int(world_size)

    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        raise RuntimeError("MASTER_ADDR not set by platform")

    master_port = int(os.environ.get("MASTER_PORT", 30333))

    return Runtime(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        model_path=ns.model_path,
        model_name=ns.model_name,
        data_path=ns.data_path,
        log_root=Path(ns.log_root).resolve(),
        experiments_json=Path(ns.experiments_json).resolve()
    )


def main() -> int:
    rt = parse_args()
    g = GlobalConfig()
    exps = load_experiments(rt.experiments_json, g)

    ts = time.strftime("%m-%d-%H-%M")
    log_dir = rt.log_root / ts
    signal_root = log_dir / "signals"
    log_dir.mkdir(parents=True, exist_ok=True)
    signal_root.mkdir(parents=True, exist_ok=True)

    for exp in tqdm(exps, desc="Experiments"):
        if not exp.enable: continue

        configs = exp.configs or g.configs
        mcs = [int(x) for x in (exp.bench_mcs or g.bench_mcs)]
        exp_signal_dir = signal_root / exp.name
        exp_signal_dir.mkdir(parents=True, exist_ok=True)

        for idx, cfg in enumerate(configs, 1):
            tp, ep = [int(x) for x in cfg.split(":", 1)]
            if rt.rank == 0:
                run_rank0_one_config(rt, g, log_dir / exp.name, exp_signal_dir, idx, tp, ep, mcs)
            else:
                run_worker_one_config(rt, log_dir / exp.name, exp_signal_dir, idx, tp, ep)

        if rt.rank == 0:
            send_signal(exp_signal_dir / "exp_done")
            for r in range(1, rt.world_size):
                wait_signal(exp_signal_dir / f"exp_ack_rank{r}", timeout_s=600)
        else:
            wait_signal(exp_signal_dir / "exp_done", timeout_s=600)
            send_signal(exp_signal_dir / f"exp_ack_rank{rt.rank}")

    send_signal(signal_root / f"abort_rank{rt.rank}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
