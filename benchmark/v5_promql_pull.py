#!/usr/bin/env python3
"""
V5: 拉取每个 sweep run 的 PromQL 数据，把 vLLM/SGLang 服务端指标
按 [wall_start_ts, wall_end_ts] 时间窗切出来存到 result.json 同名 _prom.json。
+ SGLang retract counter: 从 docker logs 里 grep "retract"。

用法:
  python3 v5_promql_pull.py --result-dir results/p1_5_chunked_prefill \
                            --prom http://192.168.1.101:30900 \
                            --engine vllm \
                            --container vllm-v5-72b-tp8

输出: 每个 RV5*_result.json 旁边生成 RV5*_prom.json
{
  "kv_usage_max": 0.97,
  "kv_usage_p95": 0.91,
  "preemptions_delta": 12,           # vLLM 用 num_preemptions_total 差分
  "retract_lines": 12,               # SGLang 用 docker logs grep "retract" 计数
  "running_max": 96,
  "waiting_max": 28,
  "decode_p99_ms": 412.3,
  "ttft_p99_ms": 6500.0,
  "queue_p99_ms": 1200.0,
}
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path
import urllib.parse, urllib.request

def prom_query(prom_url, query, ts):
    url = f"{prom_url}/api/v1/query?query={urllib.parse.quote(query)}&time={ts}"
    return json.loads(urllib.request.urlopen(url, timeout=10).read())

def prom_query_range(prom_url, query, start, end, step=5):
    url = f"{prom_url}/api/v1/query_range?query={urllib.parse.quote(query)}&start={start}&end={end}&step={step}"
    return json.loads(urllib.request.urlopen(url, timeout=15).read())

def first_value(prom_resp, default=None):
    res = prom_resp.get("data", {}).get("result", [])
    if not res: return default
    return float(res[0]["value"][1])

def range_max(prom_resp, default=None):
    res = prom_resp.get("data", {}).get("result", [])
    if not res: return default
    vals = [float(v[1]) for v in res[0]["values"]]
    return max(vals) if vals else default

def range_quantile(prom_resp, q=0.95, default=None):
    res = prom_resp.get("data", {}).get("result", [])
    if not res: return default
    vals = sorted(float(v[1]) for v in res[0]["values"])
    if not vals: return default
    idx = int(len(vals) * q)
    return vals[min(idx, len(vals)-1)]

def grep_retract(container, since_iso):
    """count 'retract' lines in docker logs since wall_start"""
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", since_iso, container],
            capture_output=True, text=True, timeout=10
        )
        return out.stdout.lower().count("retract") + out.stderr.lower().count("retract")
    except Exception:
        return None

def pull_one(result_path, prom_url, engine, container=None):
    with open(result_path) as f:
        result = json.load(f)
    summary = result.get("summary", result)
    start = summary["wall_start_ts"]
    end = summary["wall_end_ts"]
    out = {"engine": engine, "wall_start_ts": start, "wall_end_ts": end}

    if engine == "vllm":
        kv_q = "vllm:kv_cache_usage_perc"
        run_q = "vllm:num_requests_running"
        wait_q = "vllm:num_requests_waiting"
        preempt_q_start = f"vllm:num_preemptions_total @ {start}"
        preempt_q_end = f"vllm:num_preemptions_total @ {end}"
        decode_p99 = f'1000*histogram_quantile(0.99, sum(rate(vllm:request_decode_time_seconds_bucket[60s])) by (le))'
        ttft_p99 = f'1000*histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[60s])) by (le))'
        queue_p99 = f'1000*histogram_quantile(0.99, sum(rate(vllm:request_queue_time_seconds_bucket[60s])) by (le))'
    elif engine == "sglang":
        kv_q = "sglang:token_usage"
        run_q = "sglang:num_running_reqs"
        wait_q = "sglang:num_queue_reqs"
        decode_p99 = f'1000*histogram_quantile(0.99, sum(rate(sglang:time_per_output_token_seconds_bucket[60s])) by (le))'
        ttft_p99 = f'1000*histogram_quantile(0.99, sum(rate(sglang:time_to_first_token_seconds_bucket[60s])) by (le))'
        queue_p99 = None
    else:
        sys.exit(f"unknown engine {engine}")

    out["kv_usage_max"] = range_max(prom_query_range(prom_url, kv_q, start, end))
    out["kv_usage_p95"] = range_quantile(prom_query_range(prom_url, kv_q, start, end), 0.95)
    out["running_max"] = range_max(prom_query_range(prom_url, run_q, start, end))
    out["waiting_max"] = range_max(prom_query_range(prom_url, wait_q, start, end))
    out["decode_p99_ms"] = range_max(prom_query_range(prom_url, decode_p99, start, end))
    out["ttft_p99_ms"] = range_max(prom_query_range(prom_url, ttft_p99, start, end))
    if queue_p99:
        out["queue_p99_ms"] = range_max(prom_query_range(prom_url, queue_p99, start, end))

    # preemption: vllm 差分 counter, sglang 数 retract log 行
    if engine == "vllm":
        try:
            v_end = first_value(prom_query(prom_url, "vllm:num_preemptions_total", end))
            v_start = first_value(prom_query(prom_url, "vllm:num_preemptions_total", start))
            out["preemptions_delta"] = (v_end or 0) - (v_start or 0)
        except Exception as e:
            out["preemptions_delta"] = None
            out["preemptions_err"] = str(e)
    elif engine == "sglang" and container:
        # convert wall_start_ts (epoch float) to ISO-like for docker --since
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
        out["retract_lines"] = grep_retract(container, iso)

    out_path = result_path.parent / (result_path.stem.replace("_result", "") + "_prom.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {out_path.name}: kv_max={out.get('kv_usage_max'):.2%} preempt={out.get('preemptions_delta', out.get('retract_lines', 'n/a'))} run_max={out.get('running_max')}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--prom", default="http://192.168.1.101:30900")
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    ap.add_argument("--container", default=None)
    ap.add_argument("--pattern", default=None, help="glob for result files, default: RV5*<ENGINE>*_result.json")
    args = ap.parse_args()

    pat = args.pattern or f"RV5*{args.engine.upper()}*_result.json"
    files = sorted(Path(args.result_dir).glob(pat))
    print(f"found {len(files)} result files matching {pat}")
    for f in files:
        print(f"\n[{f.name}]")
        pull_one(f, args.prom, args.engine, args.container)

if __name__ == "__main__":
    main()
