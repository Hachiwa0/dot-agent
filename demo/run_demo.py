"""演示：用桩客户端跑通三类典型请求的路由全流程。

  样例1 选项题（MMLU 型）    → L1 高置信 QA → 快速通道本地直答
  样例2 数学应用题（GSM8K 型）→ L1 REASONING → 分解 → 依赖图 → 子任务级路由
  样例3 隐性工具任务          → L1 置信不足 → L2 兜底 TOOLUSE
运行: python demo/run_demo.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent import AgentPipeline  # noqa: E402
from dot_agent.gateway.stub import StubModelClient  # noqa: E402
from dot_agent.types import Caller, ModelTier  # noqa: E402

DECOMPOSE_OUTPUT = """1. 求分组数
2. 求教室面积
3. 汇总两个结果"""
DEP_OUTPUT = "s1 -> s3\ns2 -> s3"
DECOMPOSE_TOOL = """1. 查询明天天气
2. 发送提醒给同事"""
DEP_OUTPUT_TOOL = "s1 -> s2"


def _task_text(prompt: str) -> str:
    lines = [l for l in prompt.splitlines() if l.startswith("任务:")]
    return lines[-1] if lines else prompt


def local_behavior(prompt: str, caller: Caller) -> str | None:
    """按 prompt 特征模拟本地小模型的不同角色输出。"""
    if caller is Caller.CLASSIFY:
        # 模拟语义分类：隐性工具任务（无 schema）才是 L2 的主战场
        task = _task_text(prompt)
        return "TOOLUSE" if ("天气" in task or "提醒" in task) else "REASONING"
    if "依赖关系" in prompt:
        return DEP_OUTPUT_TOOL if "天气" in prompt else DEP_OUTPUT
    if "分解为若干" in prompt:
        return DECOMPOSE_TOOL if "天气" in prompt else DECOMPOSE_OUTPUT
    if "原任务" in prompt:
        return "6组; 48平方米" if "面积" in prompt else "已完成查询与提醒"
    return "42"


def cloud_behavior(prompt: str, caller: Caller) -> str | None:
    return f"[云端解答] {prompt.splitlines()[0][:30]}... 答案: 42"


async def main() -> None:
    # token_probs_mode="uniform" 模拟简单任务（α-quantile 高 → 偏本地路由）
    local = StubModelClient(ModelTier.MD, "slm-4b-stub", behavior=local_behavior,
                            token_probs_mode="uniform")
    cloud = StubModelClient(ModelTier.MC, "cloud-stub", behavior=cloud_behavior)
    pipeline = AgentPipeline(local, cloud)

    queries = [
        "下列哪项不是哺乳动物？\nA. 鲸鱼\nB. 鲨鱼\nC. 蝙蝠\nD. 海豚",          # 快速通道
        "一个班30人, 每5人一组共多少组? 教室长8米宽6米面积是多少?",               # 分解+依赖图
        "帮我查询明天的天气并发送提醒给同事",                                    # L2 兜底 TOOLUSE
    ]

    for i, q in enumerate(queries, 1):
        result = await pipeline.run(q)
        print(f"\n{'='*72}\n[样例{i}] {q[:40]}...")
        for line in result.trace:
            print(f"  │ {line}")
        print(f"  ▶ 路径={result.path}  类型={result.label.value}")
        print(f"  ▶ 答案: {result.answer[:60]}")
        m = result.metrics
        print(f"  ▶ 指标: C_time={m['C_time_s']}s  云端调用={m['cloud_calls']}  "
              f"C_API={m['C_API_total']} tok (in {m['C_API_in']}/out {m['C_API_out']})  "
              f"本地调用={m['local_calls']}")
        print(f"  ▶ 阶段耗时: {m['stage_latency_s']}")

    # 缓存命中演示：重跑样例1（deterministic 域直接命中）
    again = await pipeline.run(queries[0])
    print(f"\n{'='*72}\n[缓存验证] 重跑样例1 → 路径={again.path}")
    print(f"  缓存统计: {pipeline.cache.stats}")


if __name__ == "__main__":
    asyncio.run(main())
