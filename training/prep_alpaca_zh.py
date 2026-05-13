#!/usr/bin/env python3
"""
准备 Alpaca-zh 数据集：直接读 ModelScope cache 里的 CSV (绕过 ModelScope SDK 跟 datasets 版本兼容问题)
"""
import csv
import json
from pathlib import Path

CSV_FILE = Path("/home/liuguangli/.cache/modelscope/hub/datasets/downloads/"
                "ee3959cc16ee530c43270b123e2d8694a153a70d1b9a10d1e697df701b3fd791")
OUT_DIR = Path("/home/liuguangli/learn-ai-infra/training/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 5000  # 5k 子集
OUT_FILE = OUT_DIR / f"alpaca_zh_{N}.jsonl"

# CSV 字段: instruction, input, output
csv.field_size_limit(1024 * 1024 * 10)  # 防长 output 超 default 限制

written = 0
with open(CSV_FILE, "r", encoding="utf-8") as f_in, open(OUT_FILE, "w") as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if not instr or not out:
            continue
        f_out.write(json.dumps({"instruction": instr, "input": inp, "output": out}, ensure_ascii=False) + "\n")
        written += 1
        if written >= N:
            break

print(f"wrote {written} examples to {OUT_FILE}")
# sample
with open(OUT_FILE) as f:
    first = json.loads(f.readline())
    print(f"\nsample:")
    print(f"  instruction: {first['instruction'][:80]}")
    print(f"  input:       {first['input'][:80] if first['input'] else '(empty)'}")
    print(f"  output:      {first['output'][:80]}")
