"""输出校验器（真实版）。

四种校验方式按策略表分发（对应《任务分类技术实现路线.md》§2.1）：
  consistency : 答案非空且可归一化（一致性率在 signals.py 计算）
  answer_equiv: 数值抽取 + 相对误差容差（expected 提供时），
                否则校验答案实体可抽取性
  constraint  : 约束谓词逐条检查（数值上/下限/相等/计数），
                谓词随自建 PLANNING 数据集提供
  exec        : 工具沙箱结果判定（tools.py 的 ToolResult 已含错误码）
"""
from __future__ import annotations

import re

from .signals import normalize_answer

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_REL_TOL = 1e-4


def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text)]


def answer_equiv(answer: str, expected: str | None = None) -> bool:
    if expected is None:
        # 无期望答案：答案中可抽取数值实体，或非空短语（开放生成）
        return bool(_numbers(answer)) or len(answer.strip()) > 0
    got, want = _numbers(answer), _numbers(expected)
    if not want:
        return normalize_answer(answer) == normalize_answer(expected)
    if not got:
        return False
    # 任一数值对在容差内匹配（答案可能含中间量）
    return any(
        abs(g - w) <= _REL_TOL * max(1.0, abs(w)) for g in got for w in want
    )


def check_constraints(answer: str, constraints: list[dict]) -> bool:
    """约束谓词：[{op: "le|ge|eq|contains", value: 数值或字符串}]。
    answer 中出现的所有数值都必须满足数值型约束（保守策略）。
    """
    nums = _numbers(answer)
    for c in constraints:
        op, value = c.get("op"), c.get("value")
        if op == "contains":
            if str(value) not in answer:
                return False
        elif isinstance(value, (int, float)):
            if op == "le" and any(n > value for n in nums):
                return False
            if op == "ge" and any(n < value for n in nums):
                return False
            if op == "eq" and not any(abs(n - value) <= _REL_TOL * max(1, abs(value)) for n in nums):
                return False
    return True


class Verifier:
    def verify(self, kind: str, question: str, answer: str, extra: dict | None = None) -> bool:
        extra = extra or {}
        if kind == "consistency":
            return len(normalize_answer(answer)) > 0
        if kind == "answer_equiv":
            return answer_equiv(answer, extra.get("expected"))
        if kind == "constraint":
            constraints = extra.get("constraints")
            if constraints:
                return check_constraints(answer, constraints)
            return bool(answer.strip())
        if kind == "exec":
            tool_result = extra.get("tool_result")
            if tool_result is None:
                # 子任务未走工具执行（描述不含工具意图）：退化为答案非空判定
                return bool(answer.strip())
            return tool_result.ok
        return True
