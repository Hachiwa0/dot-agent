"""输出校验器（骨架）。

四种校验方式按策略表分发；框架阶段实现轻量版，接口固定：
  verify(kind, question, answer, extra) -> bool
真实版对照《任务分类技术实现路线.md》§2.1：
  consistency : k=3 一致性（在 signals.py 完成，此处复判最终答案可抽取性）
  answer_equiv: SymPy 表达式等价 / 数值容差比较
  constraint  : 约束谓词库逐条检查（随自建数据集交付）
  exec        : 工具沙箱执行校验（错误码分类 ARG_INVALID/NOT_FOUND/TIMEOUT）
"""
from __future__ import annotations

from ..orchestrator.signals import normalize_answer


class Verifier:
    def verify(self, kind: str, question: str, answer: str, extra: dict | None = None) -> bool:
        extra = extra or {}
        if kind == "consistency":
            # 一致率在信号阶段已算；此处校验最终答案非空且可归一化
            return len(normalize_answer(answer)) > 0
        if kind == "answer_equiv":
            # 骨架版：能抽取到数值实体即通过；真实版接期望答案做容差比较
            return normalize_answer(answer).startswith(("num:", "opt:")) or len(answer.strip()) > 0
        if kind == "constraint":
            # TODO: 约束谓词库（自建 PLANNING 数据集随附机器可查表示）
            return True
        if kind == "exec":
            # TODO: 工具沙箱（mock 环境 + 错误码分类）
            return True
        return True
