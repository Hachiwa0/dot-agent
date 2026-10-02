"""L1 规则特征层单测：四类判定 + 不确定送 L2。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.classifier.l1_rules import classify_l1, extract_features  # noqa: E402
from dot_agent.types import Label  # noqa: E402


def test_options_qa():
    f = extract_features("下列哪项不是哺乳动物？\nA. 鲸\nB. 鲨鱼\nC. 蝙蝠\nD. 海豚")
    label, conf = classify_l1(f)
    assert label is Label.QA and conf >= 0.9


def test_math_reasoning():
    f = extract_features("一个班30人, 每5人一组, 共多少组?")
    label, conf = classify_l1(f)
    assert label is Label.REASONING and conf >= 0.8


def test_constraints_planning():
    f = extract_features("预算500以内, 周六出发, 并且每天不超过3个景点, 安排行程")
    label, conf = classify_l1(f)
    assert label is Label.PLANNING and conf >= 0.8


def test_tool_schema_tooluse():
    f = extract_features('调用工具: {"tools": [{"type": "function"}]} 查天气')
    label, conf = classify_l1(f)
    assert label is Label.TOOLUSE and conf >= 0.9


def test_implicit_tool_low_conf_to_l2():
    # 无 schema 的隐性工具任务：置信 0.80 < 0.90 → 交 L2
    f = extract_features("帮我查询明天的天气并发送提醒")
    label, conf = classify_l1(f)
    assert label is Label.TOOLUSE and conf == 0.80


def test_short_question_qa():
    f = extract_features("水的沸点是多少摄氏度")
    label, conf = classify_l1(f)
    assert label is Label.QA and conf >= 0.9


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all passed")
