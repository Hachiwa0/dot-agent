"""模型预热（吸收自团队 cloud-edge-agent 的实测经验）。

冷启动首调含模型加载（实测 37.3s），预热后同规模 prompt 仅 0.04s。
不做 warmup，真实模型下的 C_time P95 会被加载时间严重污染。
stub 模式跳过（无加载成本）。
"""
from __future__ import annotations

import sys
import time

from ..types import Caller
from .base import ModelClient


async def warmup(client: ModelClient, rounds: int = 1) -> float:
    t0 = time.perf_counter()
    for i in range(rounds):
        try:
            r = await client.generate(
                "hi", max_tokens=1, temperature=0.0, caller=Caller.WARMUP
            )
            if getattr(r, "usage_missing", False):
                print(f"[gateway] ⚠ warmup 未返回 usage: {client.name}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 —— 预热失败不阻断，正式调用再暴露
            print(f"[gateway] ⚠ warmup({client.name}) 第{i + 1}轮失败: {e}", file=sys.stderr)
    return time.perf_counter() - t0
