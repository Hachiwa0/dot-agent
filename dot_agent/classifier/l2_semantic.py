"""L2 语义分类层（路线 A：本地 SLM 单次 few-shot 分类）。

v2 期可替换为 embedding + MLP 分类头（与难度 Adapter / 语义缓存共享编码器），
接口保持 classify(query) -> Label 不变。

max_tokens=4、temperature=0：标签 1 token 即止。解析失败默认 REASONING
（中性路径，走完整流水线保正确性下限）。
"""
from __future__ import annotations

from ..gateway.base import ModelClient
from ..types import Caller, Label

DEFAULT_FEWSHOT = """示例:
"帮我查明天北京的天气并设置提醒" -> TOOLUSE
"一个班级30人, 每5人一组, 多少组?" -> REASONING
"预算500内, 周六出发, 安排两日行程且每天不超过3个景点" -> PLANNING
"水的沸点是多少摄氏度" -> QA
"给三个朋友各发一条生日祝福并预订蛋糕" -> TOOLUSE
"长8米宽6米的教室, 铺每块0.5平方米的地砖需要多少块" -> REASONING"""

CLASSIFY_PROMPT = """判断以下用户任务的类型，只输出一个标签。
QA: 单跳事实或常识问答
REASONING: 需要多步推导或数学计算
PLANNING: 在多个约束下做规划或组合决策
TOOLUSE: 需要调用外部工具或执行操作

{fewshot}

任务: {query}
标签:"""


class SemanticClassifier:
    def __init__(self, client: ModelClient, fewshot: str = DEFAULT_FEWSHOT) -> None:
        self.client = client
        self.fewshot = fewshot

    async def classify(self, query: str) -> Label:
        prompt = CLASSIFY_PROMPT.format(fewshot=self.fewshot, query=query)
        result = await self.client.generate(
            prompt, max_tokens=4, temperature=0.0, caller=Caller.CLASSIFY
        )
        text = result.text.strip().upper()
        for label in Label:
            if label.value in text:
                return label
        return Label.REASONING  # 解析失败兜底：中性路径
