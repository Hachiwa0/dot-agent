"""模型网关抽象接口。

后续接入真实后端时：
  - 本地：Ollama / llama.cpp server（需验证能否返回 token logprobs，
    不能则 token_probs 传 None，α-quantile 退化为一致性信号）
  - 云端：任意 OpenAI 兼容接口（DeepSeek/GLM/Qwen 等），仅此处替换。

所有真实客户端替换 StubModelClient 后，调度层与流水线代码零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Caller, GenerationResult, ModelTier


class ModelClient(ABC):
    tier: ModelTier
    name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        caller: Caller = Caller.EXECUTE,
    ) -> GenerationResult:
        """单轮生成。真实实现负责：
        1. 记录时延（本地推理 / 网络往返分段，当前合并为 latency_s）；
        2. 返回 token 概率（若后端支持）；
        3. token 计数用后端 usage 字段，不要用估算。
        """
