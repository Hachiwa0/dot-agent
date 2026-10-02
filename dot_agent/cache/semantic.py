"""语义缓存（骨架）。

框架阶段实现 exact-match + 命中统计；embedding 相似度检索为占位接口，
后续接 bge-small-zh（与 L2 分类头 / 难度 Adapter 共享编码器）。
命中管理按任务域 + 模型版本 + TTL；仅确定性域（deterministic）允许
命中——非确定性任务永远 miss。误命中率与命中组质量差由评测器报告。
"""
from __future__ import annotations

import time


class SemanticCache:
    def __init__(self, enabled: bool = True, ttl_s: float = 3600.0) -> None:
        self.enabled = enabled
        self.ttl_s = ttl_s
        self._store: dict[str, tuple[str, float]] = {}  # key -> (answer, ts)
        self.stats = {"hit": 0, "miss": 0}

    def get(self, query: str, deterministic: bool) -> str | None:
        if not self.enabled or not deterministic:
            self.stats["miss"] += 1
            return None
        entry = self._store.get(query)
        if entry and time.time() - entry[1] < self.ttl_s:
            self.stats["hit"] += 1
            return entry[0]
        self.stats["miss"] += 1
        return None

    def put(self, query: str, answer: str, deterministic: bool) -> None:
        if self.enabled and deterministic:
            self._store[query] = (answer, time.time())

    # TODO: def similar(self, query: str) -> str | None —— 接入 embedding 检索
    # TODO: 域/模型版本键隔离、误命中审计抽样
