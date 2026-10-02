"""云端 OpenAI 兼容客户端（真实后端）。

适用 DeepSeek / GLM / Qwen(DashScope) / Kimi / OpenAI 等任意
OpenAI 兼容 chat/completions 接口。token 计数优先取响应 usage 字段。

环境变量：
  CLOUD_BASE_URL  默认 https://api.deepseek.com/v1
  CLOUD_API_KEY   必填（仅环境变量/本地 .env，不写入代码与日志）
  CLOUD_MODEL     默认 deepseek-chat
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..types import Caller, GenerationResult, ModelTier
from .base import ModelClient


class OpenAICompatClient(ModelClient):
    tier = ModelTier.MC

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("CLOUD_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("CLOUD_API_KEY", "")
        self.model = model or os.environ.get("CLOUD_MODEL", "deepseek-chat")
        self.timeout_s = timeout_s
        self.name = f"cloud:{self.model}"

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        caller: Caller = Caller.EXECUTE,
    ) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("未设置 CLOUD_API_KEY，云端客户端不可用")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"云端接口错误 {e.code}: {e.read()[:200]!r}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"云端接口连接失败: {e}") from e
        latency = time.perf_counter() - t0

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenerationResult(
            text=text,
            token_probs=None,  # 云端不做难度信号（省 token），路由信号只用本地
            prompt_tokens=usage.get("prompt_tokens") or max(1, int(len(prompt) / 1.6)),
            completion_tokens=usage.get("completion_tokens") or max(1, int(len(text) / 1.6)),
            latency_s=latency,
        )
