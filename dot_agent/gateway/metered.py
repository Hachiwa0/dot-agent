"""计量装饰客户端：包装任意 ModelClient，所有调用自动写入 CallMeter。

计量下沉到网关层，保证本地/云端、规划/校验/重试全部调用无一漏记
（申请书口径：云端 token 含规划、校验、重试；本地耗时计入 C_time）。
"""
from __future__ import annotations

from ..types import Caller, GenerationResult, ModelTier
from .base import ModelClient
from .meter import CallMeter


class MeteredClient(ModelClient):
    def __init__(self, inner: ModelClient, meter: CallMeter) -> None:
        self.inner = inner
        self.meter = meter
        self.tier = inner.tier
        self.name = inner.name

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        caller: Caller = Caller.EXECUTE,
    ) -> GenerationResult:
        result = await self.inner.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, caller=caller
        )
        self.meter.log_call(
            caller=caller,
            tier=self.tier,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=result.latency_s,
        )
        return result
