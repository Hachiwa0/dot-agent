"""客户端工厂：按环境自动装配 stub / 真实客户端。

默认（无任何环境变量）→ 桩客户端，demo 与前端零模型可跑；
设置 OLLAMA_HOST 且 Ollama 可达 → 本地真实客户端；
设置 CLOUD_API_KEY → 云端真实客户端。
两端独立降级：真实不可用时自动回落桩并给出警告，保证系统可演示。
"""
from __future__ import annotations

import os
import sys

from ..types import ModelTier
from .base import ModelClient
from .demo_behavior import cloud_demo_behavior, local_demo_behavior
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient
from .stub import StubModelClient


def _ollama_alive(host: str) -> bool:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as resp:
            return bool(json.loads(resp.read().decode()))
    except Exception:
        return False


def make_clients(
    local: ModelClient | None = None, cloud: ModelClient | None = None
) -> tuple[ModelClient, ModelClient, dict]:
    """返回 (local_client, cloud_client, runtime_info)。"""
    info: dict = {"local": "stub", "cloud": "stub", "notes": [], "warmup_needed": False}

    if local is None:
        if os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_MODEL"):
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if _ollama_alive(host):
                local = OllamaClient()
                info["local"] = f"ollama:{local.model}"
                info["warmup_needed"] = True
            else:
                info["notes"].append(f"OLLAMA 环境变量已设但 {host} 不可达，本地回落 stub")
        else:
            info["notes"].append("未设置 OLLAMA_HOST，本地使用 stub")
    else:
        info["local"] = local.name

    if cloud is None:
        if os.environ.get("CLOUD_API_KEY"):
            cloud = OpenAICompatClient()
            info["cloud"] = f"openai-compat:{cloud.model}"
        else:
            info["notes"].append("未设置 CLOUD_API_KEY，云端使用 stub")
    else:
        info["cloud"] = cloud.name

    return (
        local or StubModelClient(ModelTier.MD, "slm-stub", behavior=local_demo_behavior),
        cloud or StubModelClient(ModelTier.MC, "cloud-stub", behavior=cloud_demo_behavior),
        info,
    )


async def warmup_real(info: dict, local: ModelClient, cloud: ModelClient) -> None:
    """真实客户端预热（stub 跳过）。server/eval 启动时调用，防冷启动污染 P95。"""
    from .warmup import warmup

    if info.get("warmup_needed"):
        dt = await warmup(local)
        print(f"[gateway] 本地模型预热完成 {dt:.1f}s")
    if info.get("cloud", "").startswith("openai-compat"):
        dt = await warmup(cloud)
        print(f"[gateway] 云端预热完成 {dt:.1f}s")


def report(info: dict) -> None:
    for note in info.get("notes", []):
        print(f"[gateway] {note}", file=sys.stderr)
