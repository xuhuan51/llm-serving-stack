#!/usr/bin/env python3
"""
V5: 双引擎 sweep 出对比图。读 RV5*_result.json + RV5*_prom.json，
出 4 张图：
  F1: throughput vs concurrency  (vLLM vs SGLang)
  F2: P99 TPOT/ITL vs concurrency (vLLM vs SGLang)
  F3: KV usage max% + preemption count vs concurrency (vLLM vs SGLang)
  F4: TTFT P99 vs concurrency (vLLM vs SGLang)
"""
import argparse, json, re
from pathlib import Path
import matplotlib.pyplot as plt

def load_engine(result_dir, engine):
    """ return list of (c, client_data, prom_data) sorted by c """
    pat_tag = engine.upper()
    out = []
    for rf in sorted(Path(result_dir).glob(f"RV5{pat_tag}_TP8_C*_result.json")):
        with open(rf) as f:
            client = json.load(f)
        # extract c from filename: RV5VLLM_TP8_C32_result.json -> 32
        c = int(re.search(r"_C(\d+)", rf.stem).group(1))
        prom_path = rf.parent / rf.stem.replace("_result", "_prom.json")
        prom = {}
        if prom_path.exists():
            with open(prom_path) as f:
                prom = json.load(f)
        out.append((c, client, prom))
    return out

def get(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/liuguangli/learn-ai-infra/serving-benchmark/results/p1_5_chunked_prefill")
    ap.add_argument("--out", default=None, help="output dir for figures")
    args = ap.parse_args()
    out_dir = Path(args.out or args.dir) / "figures"
    out_dir.mkdir(exist_ok=True)

    vllm = load_engine(args.dir, "vllm")
    sglang = load_engine(args.dir, "sglang")
    print(f"vLLM points: {[c for c,_,_ in vllm]}")
    print(f"SGLang points: {[c for c,_,_ in sglang]}")

    def plot(name, title, ylabel, vfn, sfn, log_y=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        if vllm:
            xs = [c for c,_,_ in vllm]
            ys = [vfn(c, cl, pr) for c, cl, pr in vllm]
            ax.plot(xs, ys, "o-", label="vLLM", linewidth=2, markersize=8)
        if sglang:
            xs = [c for c,_,_ in sglang]
            ys = [sfn(c, cl, pr) for c, cl, pr in sglang]
            ax.plot(xs, ys, "s-", label="SGLang", linewidth=2, markersize=8)
        ax.set_xlabel("concurrency")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ticks = sorted(set([c for c,_,_ in vllm] + [c for c,_,_ in sglang]))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        if log_y: ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        path = out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        print(f"  saved {path}")
        plt.close(fig)

    # F1 throughput
    plot("F1_throughput",
         "Token throughput vs concurrency  (Qwen2.5-72B-AWQ TP=8 PCIe)",
         "throughput (tok/s)",
         lambda c, cl, pr: cl.get("summary", cl).get("tok_per_sec", 0),
         lambda c, cl, pr: cl.get("summary", cl).get("tok_per_sec", 0))

    # F2 P99 inter-token latency
    plot("F2_p99_itl",
         "P99 inter-token latency vs concurrency",
         "P99 ITL (ms)",
         lambda c, cl, pr: get(pr, "decode_p99_ms", default=None) or cl.get("summary", cl).get("tpot", {}).get("p99", 0)*1000,
         lambda c, cl, pr: get(pr, "decode_p99_ms", default=None) or cl.get("summary", cl).get("tpot", {}).get("p99", 0)*1000)

    # F3a KV usage
    plot("F3a_kv_usage",
         "Max KV cache usage % vs concurrency",
         "KV cache usage (max %)",
         lambda c, cl, pr: (get(pr, "kv_usage_max", default=0) or 0) * 100,
         lambda c, cl, pr: (get(pr, "kv_usage_max", default=0) or 0) * 100)

    # F3b preemption count
    plot("F3b_preemption",
         "Preemption events per run (n=500 requests)",
         "preemption count (Δ counter for vLLM, log-grep for SGLang)",
         lambda c, cl, pr: get(pr, "preemptions_delta", default=0) or 0,
         lambda c, cl, pr: get(pr, "retract_lines", default=0) or 0)

    # F4 TTFT P99
    plot("F4_ttft",
         "P99 TTFT vs concurrency",
         "P99 TTFT (ms)",
         lambda c, cl, pr: get(pr, "ttft_p99_ms", default=None) or cl.get("summary", cl).get("ttft", {}).get("p99", 0)*1000,
         lambda c, cl, pr: get(pr, "ttft_p99_ms", default=None) or cl.get("summary", cl).get("ttft", {}).get("p99", 0)*1000)

if __name__ == "__main__":
    main()
