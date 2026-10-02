"""L1 规则特征层：零推理成本的文本特征提取 + 三态判定。

三态输出：
  (LABEL, conf≥0.9)  → 高置信：QA 走快速通道，其余直达策略表（跳过 L2 省一次推理）
  (None, 0.0)        → 送 L2 语义分类
词表与置信度外置 config/classify_rules.json，便于在评测集上扫描调优。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..types import Label, TaskFeatures

DEFAULT_RULES: dict = {
    "tool_schema_markers": ['"tools"', '"type": "function"', "```json", "tool_call", "def "],
    "action_verbs": ["查询", "预订", "发送", "检索", "调用", "打开", "设置", "提醒", "下单", "导出", "搜索", "订阅"],
    "constraint_markers": ["同时", "并且", "不超过", "至少", "最多", "预算", "优先", "避免", "以内", "以上"],
    "question_words": ["是什么", "什么是", "为什么", "是谁", "哪个", "哪些", "多少", "什么时候", "在哪里"],
    "deterministic_words": ["多少", "计算", "求", "等于"],
    "arith_words": ["每", "共", "倍", "平均", "相差", "合计", "面积", "总数"],
    "short_query_max_tokens": 200,
}

_OPTIONS_RE = re.compile(r"(?m)^\s*[A-D][\.、\)]")
_MATH_RE = re.compile(r"\d+\s*[+\-*/^=×÷]\s*\d")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def load_rules(path: str | Path | None = None) -> dict:
    if path is None:
        rule_file = Path(__file__).resolve().parents[2] / "config" / "classify_rules.json"
        if rule_file.exists():
            return json.loads(rule_file.read_text(encoding="utf-8"))
        return dict(DEFAULT_RULES)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_features(query: str, rules: dict | None = None) -> TaskFeatures:
    r = rules or DEFAULT_RULES
    n_tokens = max(1, int(len(query) / 1.6))
    return TaskFeatures(
        has_tool_schema=any(m in query for m in r["tool_schema_markers"]),
        action_verbs=sum(1 for w in r["action_verbs"] if w in query),
        constraint_markers=sum(1 for w in r["constraint_markers"] if w in query),
        has_math_expr=bool(_MATH_RE.search(query)),
        num_count=len(_NUM_RE.findall(query)),
        arith_words=sum(1 for w in r["arith_words"] if w in query),
        has_options=bool(_OPTIONS_RE.search(query)),
        question_words=sum(1 for w in r["question_words"] if w in query),
        n_tokens=n_tokens,
        deterministic=bool(_OPTIONS_RE.search(query))
        or any(w in query for w in r["deterministic_words"]),
        ctx_len_seg="0-1k" if n_tokens < 1000 else ("1k-4k" if n_tokens < 4000 else ">4k"),
    )


def classify_l1(features: TaskFeatures, rules: dict | None = None) -> tuple[Label | None, float]:
    """按优先级评分，返回 (标签或 None, 置信度)。"""
    f = features
    if f.has_options:
        return Label.QA, 0.95
    if f.has_tool_schema:
        return Label.TOOLUSE, 0.95
    if f.constraint_markers >= 2:
        return Label.PLANNING, 0.85
    # 显式运算式，或数值密度高（数字≥2 且含算术词——中文应用题常无运算符）
    if f.has_math_expr or (f.num_count >= 2 and f.arith_words >= 1):
        return Label.REASONING, 0.85
    if f.action_verbs >= 2:
        return Label.TOOLUSE, 0.80  # 无 schema 的隐性工具任务，置信不足交 L2 兜底
    r = rules or DEFAULT_RULES
    if f.question_words >= 1 and f.n_tokens <= r["short_query_max_tokens"]:
        return Label.QA, 0.90
    return None, 0.0
