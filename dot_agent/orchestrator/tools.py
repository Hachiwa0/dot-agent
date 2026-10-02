"""工具沙箱（TOOLUSE 域的"执行即分类器"）。

- ToolRegistry：注册 mock 工具（演示用，真实场景替换为绑定 API 的实现）；
- 意图解析：从子任务描述规则抽取工具与参数（框架阶段不依赖模型输出 JSON）；
- 错误码分类：OK / ARG_INVALID（参数错，可本地重试）/ NOT_FOUND（幻觉调用，
  偏云端信号）/ TIMEOUT（升级云端）——对应《任务分类技术实现路线.md》§2.1(d)。
执行反馈接入 SignalScores.exec_ok，构成 TOOLUSE 的难度信号主来源。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolResult:
    ok: bool
    output: str
    error_code: str  # OK / ARG_INVALID / NOT_FOUND / TIMEOUT


@dataclass
class Tool:
    name: str
    fn: Callable[..., str]
    arg_spec: list[str]  # 参数名列表


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, arg_spec: list[str], fn: Callable[..., str]) -> None:
        self._tools[name] = Tool(name, fn, arg_spec)

    # ---- 意图解析（规则版；可替换为模型结构化输出）----
    _INTENT_PATTERNS: list[tuple[str, re.Pattern, Callable[[re.Match], dict]]] = [
        ("get_weather", re.compile(r"(?:查询|看看|获取)(?P<city>[一-龥]{2,4}?)(?:明天的)?天气"),
         lambda m: {"city": m.group("city")}),
        ("send_reminder", re.compile(r"(?:发送|给)(?:提醒|通知)给?(?P<to>[一-龥]{2,8})"),
         lambda m: {"to": m.group("to")}),
        ("calculator", re.compile(r"计算(?P<expr>[\d+\-*/().\s]+)"),
         lambda m: {"expr": m.group("expr").strip()}),
    ]

    def parse_intent(self, description: str) -> tuple[str, dict] | None:
        for name, pattern, extract in self._INTENT_PATTERNS:
            m = pattern.search(description)
            if m:
                return name, extract(m)
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"工具 {name} 不存在", "NOT_FOUND")
        try:
            missing = [a for a in tool.arg_spec if a not in args or args[a] in (None, "")]
            if missing:
                return ToolResult(False, f"参数缺失: {missing}", "ARG_INVALID")
            return ToolResult(True, str(tool.fn(**args)), "OK")
        except ZeroDivisionError:
            return ToolResult(False, "除零错误", "ARG_INVALID")
        except Exception as e:  # 沙箱兜底：工具异常不外泄
            return ToolResult(False, f"执行异常: {e}", "ARG_INVALID")

    def try_execute(self, description: str) -> ToolResult | None:
        """从描述解析意图并执行；解析不到返回 None（非工具型子任务）。"""
        intent = self.parse_intent(description)
        if intent is None:
            return None
        return self.execute(*intent)


def default_registry() -> ToolRegistry:
    """内置 mock 工具：确定性输出，便于评测与演示。"""
    reg = ToolRegistry()

    def get_weather(city: str) -> str:
        base = {"北京": 18, "上海": 22, "广州": 27, "深圳": 26}
        return f"{city}明天晴，气温 {base.get(city, 20)}°C"

    def send_reminder(to: str) -> str:
        return f"已向 {to} 发送提醒（mock）"

    def calculator(expr: str) -> str:
        # 沙箱内只允许数字与运算符（正则白名单），杜绝 eval 注入
        if not re.fullmatch(r"[\d+\-*/().\s]+", expr):
            raise ValueError("非法表达式")
        return f"{expr} = {eval(expr, {'__builtins__': {}}, {}):.6g}"  # noqa: S307 白名单后求值

    reg.register("get_weather", ["city"], get_weather)
    reg.register("send_reminder", ["to"], send_reminder)
    reg.register("calculator", ["expr"], calculator)
    return reg
