"""评测 runner：四模式对照（对齐申请书口径）。

模式：
  local_only  纯本地基线：整题一次本地调用
  cloud_only  纯云端基线：整题一次云端调用（CoT 式，性能与成本上界参照）
  rule        规则协同 v0：L1/L2 分类 + 类型默认倾向路由（信号关闭）
  signal      信号协同 v1：完整统计信号路由（α-quantile + 一致性 + 执行反馈）

指标：Acc / C_API（仅云端 in+out，含重试）/ C_time（中位数 + P95）。
评分规则固定：num=数值容差、letter=选项字母、constraint=约束谓词、
contains=关键内容包含。

用法：
  python eval/runner.py                    # stub 模式烟测（验证流程可跑）
  python eval/runner.py --all              # 四模式全跑，输出 eval/report.md
  接真实模型：设 OLLAMA_HOST / CLOUD_API_KEY 后同命令。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from dot_agent import AgentPipeline  # noqa: E402
from dot_agent.gateway.factory import make_clients  # noqa: E402
from dot_agent.gateway.meter import CallMeter  # noqa: E402
from dot_agent.orchestrator.verify import answer_equiv, check_constraints  # noqa: E402
from dot_agent.orchestrator.signals import normalize_answer  # noqa: E402
from dot_agent.types import Caller, ModelTier  # noqa: E402

DATASETS = ["gsm8k_style", "mmlu_style", "planning_cn", "tooluse_cn"]


def load_dataset(name: str) -> list[dict]:
    rows = []
    for line in (EVAL_DIR / "datasets" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_one(item: dict, answer: str) -> bool:
    kind = item["score"]
    if kind == "num":
        return answer_equiv(answer, item["expected"])
    if kind == "letter":
        return normalize_answer(answer) == f"opt:{item['expected']}"
    if kind == "constraint":
        return check_constraints(answer, item.get("constraints", []))
    if kind == "contains":
        return str(item["expected"]) in answer
    return False


def run_baselines(items: list[dict], tier: ModelTier, client) -> list[dict]:
    """纯本地 / 纯云端基线：整题一次调用（不走流水线）。"""
    records = []
    for item in items:
        t0 = time.perf_counter()
        result = asyncio.run(
            client.generate(item["query"], max_tokens=512, temperature=0.0, caller=Caller.EXECUTE)
        )
        dt = time.perf_counter() - t0
        records.append(
            {
                "id": item["id"],
                "correct": score_one(item, result.text),
                "cloud_in": result.prompt_tokens if tier is ModelTier.MC else 0,
                "cloud_out": result.completion_tokens if tier is ModelTier.MC else 0,
                "time_s": dt,
            }
        )
    return records


def run_pipeline_mode(items: list[dict], local, cloud, use_signals: bool) -> list[dict]:
    pipeline = AgentPipeline(local, cloud, use_signals=use_signals)
    records = []
    for item in items:
        extra = {}
        if item["score"] == "constraint":
            extra["constraints"] = item.get("constraints")
        result = asyncio.run(pipeline.run(item["query"]))
        m = result.metrics
        records.append(
            {
                "id": item["id"],
                "correct": score_one(item, result.answer),
                "cloud_in": m["C_API_in"],
                "cloud_out": m["C_API_out"],
                "time_s": m["C_time_s"] or 0.0,
                "path": result.path,
            }
        )
        # 每条独立计量：基线对照需逐题记录，重置流水线 meter
        pipeline.meter = CallMeter()
        pipeline.local.meter = pipeline.meter
        pipeline.cloud.meter = pipeline.meter
    return records


def summarize(records: list[dict]) -> dict:
    n = len(records)
    times = sorted(r["time_s"] for r in records)
    p95_idx = min(int(0.95 * (n - 1)), n - 1)
    return {
        "n": n,
        "acc": sum(r["correct"] for r in records) / n if n else 0.0,
        "cloud_in": sum(r["cloud_in"] for r in records),
        "cloud_out": sum(r["cloud_out"] for r in records),
        "t_med": times[n // 2] if n else 0.0,
        "t_p95": times[p95_idx] if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="四模式全跑（默认仅烟测 signal 模式）")
    args = parser.parse_args()

    local, cloud, info = make_clients()
    print(f"运行时: local={info['local']} cloud={info['cloud']}")
    for note in info["notes"]:
        print(f"  注: {note}")
    if info["local"] == "stub":
        print("  ⚠ stub 模式——指标无实际意义，仅验证评测流程可跑通")

    all_items = [item for ds in DATASETS for item in load_dataset(ds)]
    modes: list[tuple[str, list[dict]]] = [("signal", run_pipeline_mode(all_items, local, cloud, True))]
    if args.all:
        modes += [
            ("rule", run_pipeline_mode(all_items, local, cloud, False)),
            ("local_only", run_baselines(all_items, ModelTier.MD, local)),
            ("cloud_only", run_baselines(all_items, ModelTier.MC, cloud)),
        ]

    lines = [
        "# 评测报告",
        "",
        f"运行时: local={info['local']}, cloud={info['cloud']}",
        "",
        "| 模式 | n | Acc | C_API(in/out) | C_time 中位数(s) | C_time P95(s) |",
        "|---|---|---|---|---|---|",
    ]
    for name, records in modes:
        s = summarize(records)
        lines.append(
            f"| {name} | {s['n']} | {s['acc']:.1%} | {s['cloud_in']}/{s['cloud_out']} "
            f"| {s['t_med']:.3f} | {s['t_p95']:.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (EVAL_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"已写入 {EVAL_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
