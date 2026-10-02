"""工具沙箱与校验器单测。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.orchestrator.tools import default_registry  # noqa: E402
from dot_agent.orchestrator.verify import Verifier, answer_equiv, check_constraints  # noqa: E402


def test_intent_parsing():
    reg = default_registry()
    assert reg.parse_intent("查询上海明天的天气") == ("get_weather", {"city": "上海"})
    assert reg.parse_intent("发送提醒给同事")[0] == "send_reminder"
    assert reg.parse_intent("计算 12*(3+4)")[0] == "calculator"
    assert reg.parse_intent("今天心情不错") is None


def test_tool_exec_ok_and_errors():
    reg = default_registry()
    ok = reg.try_execute("查询北京的天气")
    assert ok.ok and "北京" in ok.output and ok.error_code == "OK"
    # 参数缺失（正则只抓到城市但工具需要 city——构造直接调用）
    bad = reg.execute("get_weather", {})
    assert not bad.ok and bad.error_code == "ARG_INVALID"
    # 幻觉工具
    ghost = reg.execute("nonexistent_tool", {})
    assert not ghost.ok and ghost.error_code == "NOT_FOUND"
    # 非法表达式（沙箱白名单拦截）
    evil = reg.execute("calculator", {"expr": "__import__('os')"})
    assert not evil.ok and evil.error_code == "ARG_INVALID"


def test_answer_equiv_tolerance():
    assert answer_equiv("答案是 6.0001 组", "6")
    assert answer_equiv("共 48 平方米", "48")
    assert not answer_equiv("42", "48")


def test_constraints():
    assert check_constraints("总花费 460 元", [{"op": "le", "value": 500}])
    assert not check_constraints("总花费 620 元", [{"op": "le", "value": 500}])
    assert check_constraints("安排 4 段", [{"op": "ge", "value": 3}, {"op": "le", "value": 4}])
    assert check_constraints("含素食 6 份", [{"op": "contains", "value": "素食"}])


def test_verifier_exec_without_tool_result():
    v = Verifier()
    assert v.verify("exec", "q", "某个答案") is True      # 无工具结果退化为非空判定
    assert v.verify("exec", "q", "", {"tool_result": type("R", (), {"ok": False})()}) is False


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all passed")
