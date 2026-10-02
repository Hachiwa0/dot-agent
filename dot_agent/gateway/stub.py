"""桩客户端：框架阶段替身，接口与真实客户端完全一致。

behavior 注入方式：
    def behavior(prompt: str, caller: Caller) -> str | None
    返回 None 时使用默认文本。demo/tests 用它模拟分解输出、依赖边、
    标签、答案等，从而在无任何模型 API 的情况下跑通完整路由流程。

token_probs_mode:
    "uniform"  全 0.9 —— 模拟简单任务（α-quantile 高 → 路由本地）
    "hard"     概率随位置衰减到 0.2 —— 模拟困难任务（α-quantile 低 → 路由云端）
"""
from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable

from ..types import Caller, GenerationResult, ModelTier
from .base import ModelClient


class StubModelClient(ModelClient):
    def __init__(
        self,
        tier: ModelTier,
        name: str = "stub",
        behavior: Callable[[str, Caller], str | None] | None = None,
        base_latency_s: float = 0.005,
        token_probs_mode: str = "uniform",
    ) -> None:
        self.tier = tier
        self.name = name
        self.behavior = behavior
        self.base_latency_s = base_latency_s
        self.token_probs_mode = token_probs_mode
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        caller: Caller = Caller.EXECUTE,
    ) -> GenerationResult:
        self.call_count += 1
        t0 = time.perf_counter()
        await asyncio.sleep(self.base_latency_s)

        text = self.behavior(prompt, caller) if self.behavior else None
        if inspect.isawaitable(text):  # 允许注入 async behavior
            text = await text
        if text is None:
            text = f"[{self.name}] 默认回答"

        n_completion = min(len(text), max_tokens * 4)
        if self.token_probs_mode == "hard":
            probs = [max(0.2, 0.9 - 0.15 * i) for i in range(max(1, n_completion // 4))]
        else:
            probs = [0.9] * max(1, n_completion // 4)

        # 中文按 字符数/1.6 估算 token；真实客户端应使用后端 usage 字段
        return GenerationResult(
            text=text[: max_tokens * 4],
            token_probs=probs,
            prompt_tokens=max(1, int(len(prompt) / 1.6)),
            completion_tokens=max(1, int(n_completion / 1.6)),
            latency_s=time.perf_counter() - t0,
        )
