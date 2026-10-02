"""三指标计量器。

口径严格对齐申请书：
  C_API  : 仅云端（MC）调用的 in+out token 总和，包含规划/分解决策之外的
           校验、重试、汇总等全部云端调用；本地 token 不计入。
  C_time : 从收到请求到最终结果返回的墙钟时间（在 pipeline 记总时长），
           本地推理耗时包含在内；本类同时给出阶段拆解。
  Acc    : 由外部评测器按固定评分规则判定，本类不负责。
"""
from __future__ import annotations

import time
from collections import defaultdict

from ..types import CallLog, Caller, ModelTier


class CallMeter:
    def __init__(self) -> None:
        self.logs: list[CallLog] = []
        self._stage_marks: dict[str, float] = {}
        self.stage_latency: dict[str, float] = defaultdict(float)

    # ---- 计量 ----
    def log(self, log: CallLog) -> None:
        self.logs.append(log)

    def log_call(
        self,
        caller: Caller,
        tier: ModelTier,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        note: str = "",
    ) -> None:
        self.logs.append(
            CallLog(caller, tier, prompt_tokens, completion_tokens, latency_s, note)
        )

    # ---- 阶段计时（调度层时延拆解：分类/分解/依赖判断/执行/校验/汇总）----
    def stage(self, name: str) -> None:
        self._stage_marks[name] = time.perf_counter()

    def stage_end(self, name: str) -> float:
        dt = time.perf_counter() - self._stage_marks[name]
        self.stage_latency[name] += dt
        return dt

    # ---- 汇总 ----
    @property
    def cloud_in_tokens(self) -> int:
        return sum(l.prompt_tokens for l in self.logs if l.tier is ModelTier.MC)

    @property
    def cloud_out_tokens(self) -> int:
        return sum(l.completion_tokens for l in self.logs if l.tier is ModelTier.MC)

    def summarize(self, total_time_s: float | None = None) -> dict:
        by_caller: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens": 0})
        for l in self.logs:
            entry = by_caller[l.caller.value]
            entry["calls"] += 1
            entry["tokens"] += l.prompt_tokens + l.completion_tokens
        return {
            "C_time_s": round(total_time_s, 4) if total_time_s else None,
            "cloud_calls": sum(1 for l in self.logs if l.tier is ModelTier.MC),
            "C_API_in": self.cloud_in_tokens,
            "C_API_out": self.cloud_out_tokens,
            "C_API_total": self.cloud_in_tokens + self.cloud_out_tokens,
            "local_calls": sum(1 for l in self.logs if l.tier is ModelTier.MD),
            "local_latency_s": round(
                sum(l.latency_s for l in self.logs if l.tier is ModelTier.MD), 4
            ),
            "stage_latency_s": {k: round(v, 4) for k, v in self.stage_latency.items()},
            "by_caller": dict(by_caller),
        }
