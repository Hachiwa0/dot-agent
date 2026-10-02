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
from urllib.parse import urlsplit

from ..types import Caller, GenerationResult, ModelTier
from .base import ModelClient


def _make_opener(base_url: str):
    """回环地址显式绕过系统代理（WSL/企业代理常见坑），云端保留环境代理。"""
    from urllib.request import ProxyHandler, build_opener

    handlers = []
    if urlsplit(base_url).hostname in ("127.0.0.1", "localhost", "::1"):
        handlers.append(ProxyHandler({}))
    return build_opener(*handlers)


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
        self.opener = _make_opener(self.base_url)

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
            with self.opener.open(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"云端接口错误 {e.code}: {e.read()[:200]!r}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"云端接口连接失败: {e}") from e
        latency = time.perf_counter() - t0

        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish = choice.get("finish_reason")
        if finish not in ("stop", None):  # 吸收 edge-cloud-prototype：非正常截断标注
            text += f"\n[finish_reason={finish}]"
        usage = data.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        usage_missing = not (isinstance(pt, int) and pt >= 0 and isinstance(ct, int))
        # usage 缺失时 token 记 0 并标记——绝不按字数估算（计量红线）
        return GenerationResult(
            text=text,
            token_probs=None,  # 云端不做难度信号（省 token），路由信号只用本地
            prompt_tokens=pt or 0,
            completion_tokens=ct or 0,
            latency_s=latency,
            usage_missing=usage_missing,
        )
