"""端到端编排流水线（框架核心）。

流程（对应方案架构图）：
  缓存查询 → 分类（L1/L2）→ 快速通道或重流水线 → 汇总 → 写缓存 → 记指标

快速通道：策略 decompose=False（QA）→ 本地直答 + k=3 一致性校验。
重流水线：分解 → 依赖图 → 逐层并行执行（每子任务：信号收集 → 路由分配 →
生成 → 校验 → 失败升级，受 EscalationGuard 约束）→ 本地汇总。

L3 运行时重分类钩子：快速通道一致性不过 → 升级 REASONING 走完整流水线；
分解失败 → 整题收敛云端。全部事件写入 trace（"失败升级"统计来源）。

计量：local/cloud 均以 MeteredClient 包装，覆盖全部 caller 路径。
"""
from __future__ import annotations

import time

from .cache.semantic import SemanticCache
from .classifier.l1_rules import extract_features
from .classifier.l2_semantic import SemanticClassifier
from .classifier.pipeline import ClassificationPipeline
from .gateway.base import ModelClient
from .gateway.meter import CallMeter
from .gateway.metered import MeteredClient
from .orchestrator.decomposer import TaskDecomposer
from .orchestrator.escalator import EscalationGuard
from .orchestrator.router import POLICY, Router
from .orchestrator.scheduler import TaskScheduler
from .orchestrator.signals import SignalCollector
from .orchestrator.tools import default_registry
from .orchestrator.verify import Verifier
from .types import Caller, Label, ModelTier, RunResult, SubTask


class AgentPipeline:
    def __init__(
        self,
        local_client: ModelClient,
        cloud_client: ModelClient,
        meter: CallMeter | None = None,
        theta_default: float = 0.5,
        use_signals: bool = True,  # False → v0 规则协同（信号关闭，仅 bias 路由）
    ) -> None:
        self.use_signals = use_signals
        self.meter = meter or CallMeter()
        # 计量包装：之后所有组件只持有 MeteredClient，调用即记录
        self.local: MeteredClient = MeteredClient(local_client, self.meter)
        self.cloud: MeteredClient = MeteredClient(cloud_client, self.meter)
        self.classifier = ClassificationPipeline(SemanticClassifier(self.local))
        self.decomposer = TaskDecomposer(self.local)
        self.scheduler = TaskScheduler(self.local)
        self.router = Router(theta_default)
        self.signals = SignalCollector(self.local)
        self.verifier = Verifier()
        self.tools = default_registry()
        self.cache = SemanticCache()

    # ------------------------------------------------------------------ #
    async def run(self, query: str) -> RunResult:
        t0 = time.perf_counter()
        trace: list[str] = []
        meter = self.meter

        # 1. 语义缓存（仅确定性域可命中）
        features = extract_features(query, self.classifier.rules)
        cached = self.cache.get(query, features.deterministic)
        if cached is not None:
            trace.append(f"cache_hit（deterministic={features.deterministic}）")
            return self._finalize(query, cached, Label.QA, "cache_hit", trace, t0)

        # 2. 分类（L1 短路 / L2 兜底）
        meter.stage("classify")
        cls = await self.classifier.classify(query)
        meter.stage_end("classify")
        policy = POLICY[cls.label]
        trace.append(
            f"classify: {cls.label.value} conf={cls.conf:.2f} src={cls.source} "
            f"fast_path={cls.fast_path}（约束={features.constraint_markers} 数学式="
            f"{features.has_math_expr} 选项={features.has_options} 工具词={features.action_verbs}）"
        )

        # 3. 快速通道：decompose=False（QA 类）→ 本地直答 + 一致性校验
        if cls.fast_path or policy.decompose is False:
            answer, ok = await self._fast_path(query, policy.verify, trace)
            if ok:
                self.cache.put(query, answer, features.deterministic)
                return self._finalize(query, answer, cls.label, "fast_path", trace, t0)
            # L3 重分类：QA 一致性不过 → 升级 REASONING 走完整流水线
            trace.append("L3 重分类: QA → REASONING（一致性校验未通过）")
            from .types import Classification  # 局部导入避免与模块头部循环依赖

            cls = Classification(Label.REASONING, 0.5, "reclassify", features, False)
            policy = POLICY[cls.label]

        # 4. 重流水线（REASONING / PLANNING / TOOLUSE.dynamic）
        guard = EscalationGuard()
        meter.stage("decompose")
        subtasks = await self.decomposer.decompose(query)
        meter.stage_end("decompose")

        if subtasks is None:
            guard.converge_to_cloud()
            trace.append("分解失败 → 整题收敛云端（direct_cloud）")
            answer = (
                await self.cloud.generate(
                    query, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE
                )
            ).text
            return self._finalize(query, answer, cls.label, "direct_cloud", trace, t0)

        trace.append(
            f"分解出 {len(subtasks)} 个子任务: "
            + "; ".join(f"{st.id}:{st.description[:12]}" for st in subtasks)
        )

        meter.stage("execute")
        answers = await self.scheduler.run(
            query, subtasks, self._make_executor(cls, policy, guard, trace)
        )
        meter.stage_end("execute")

        # 5. 本地汇总（零云端成本）
        meter.stage("summarize")
        summary_prompt = (
            f"原任务: {query}\n各子问题及答案:\n"
            + "\n".join(f"{st.id}: {st.description} -> {answers[st.id]}" for st in subtasks)
            + "\n请给出最终答案。"
        )
        final = await self.local.generate(
            summary_prompt, max_tokens=256, temperature=0.0, caller=Caller.SUMMARIZE
        )
        meter.stage_end("summarize")
        trace.append("本地汇总完成")
        self.cache.put(query, final.text, features.deterministic)
        subtask_data = [
            {
                "id": st.id,
                "description": st.description,
                "deps": list(st.deps),
                "assigned": st.assigned.value if st.assigned else None,
                "level": st.level,
                "status": st.status,
                "answer": st.answer[:80],
            }
            for st in subtasks
        ]
        return self._finalize(query, final.text, cls.label, "pipeline", trace, t0, subtask_data)

    # ------------------------------------------------------------------ #
    def reset_meter(self) -> None:
        """重置计量器（服务端/评测逐请求计量用；缓存等状态保留）。"""
        self.meter = CallMeter()
        self.local.meter = self.meter
        self.cloud.meter = self.meter

    async def _fast_path(self, query: str, verify_kind: str, trace: list[str]) -> tuple[str, bool]:
        """本地直答；k=3 一致性（复用 SignalCollector 的 greedy 结果当答案）。"""
        sig, greedy = await self.signals.collect(query)
        trace.append(
            f"fast_path 信号: α-quantile={sig.alpha_quantile} "
            f"consistency={sig.consistency} → 本地直答"
        )
        ok = sig.consistency == 1.0 and self.verifier.verify(verify_kind, query, greedy.text)
        return greedy.text, ok

    def _make_executor(self, cls, policy, guard: EscalationGuard, trace: list[str]):
        label, verify_kind = cls.label, policy.verify

        async def executor(st: SubTask, ctx: dict[str, str]) -> str:
            # TOOLUSE：执行即分类器——先本地沙箱执行，错误码即难度信号
            if label is Label.TOOLUSE:
                tr = self.tools.try_execute(st.description)
                if tr is not None:
                    if tr.ok:
                        st.assigned = ModelTier.MD
                        trace.append(f"{st.id} 工具执行成功（本地沙箱）")
                        return tr.output
                    trace.append(f"{st.id} 工具执行失败[{tr.error_code}]")
                    if guard.can_escalate():
                        guard.escalate()
                        st.assigned = ModelTier.MC
                        trace.append(
                            f"{st.id} → 升级云端（guard 剩余额度 {guard.remaining}）"
                        )
                        return (
                            await self.cloud.generate(
                                st.description, max_tokens=256,
                                temperature=0.0, caller=Caller.RETRY,
                            )
                        ).text
                    trace.append(f"{st.id} 升级额度耗尽 → 保留失败输出")
                    return tr.output

            # 上下文裁剪：只注入直接前驱答案
            ctx_part = "".join(f"\n已知 {d}: {ctx[d]}" for d in st.deps if d in ctx)
            prompt = f"子问题: {st.description}{ctx_part}\n请给出简洁答案。"

            if self.use_signals:
                sig, greedy = await self.signals.collect(prompt)
            else:  # v0 规则协同：无难度信号，路由仅按类型默认倾向
                sig, greedy = None, None
            decision = self.router.assign(label, sig)
            st.assigned = decision.tier
            trace.append(f"{st.id} 路由: {decision.reason}")

            if decision.tier is ModelTier.MD:
                if greedy is None:  # 信号关闭时本地直接生成
                    greedy = await self.local.generate(
                        prompt, max_tokens=256, temperature=0.0, caller=Caller.EXECUTE
                    )
                answer = greedy.text  # 复用 greedy 试答，省一次生成
                ok = self.verifier.verify(verify_kind, st.description, answer)
            else:
                answer = (
                    await self.cloud.generate(
                        prompt, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE
                    )
                ).text
                ok = True

            if not ok and decision.tier is ModelTier.MD:
                if guard.can_escalate():
                    guard.escalate()
                    st.assigned = ModelTier.MC
                    trace.append(f"{st.id} 校验不过 → 升级云端（guard 剩余额度 {guard.remaining}）")
                    answer = (
                        await self.cloud.generate(
                            prompt, max_tokens=512, temperature=0.0, caller=Caller.RETRY
                        )
                    ).text
                else:
                    trace.append(f"{st.id} 校验不过且升级额度耗尽 → 保留本地答案")
            return answer

        return executor

    # ------------------------------------------------------------------ #
    def _finalize(self, query, answer, label, path, trace, t0, subtasks=None) -> RunResult:
        total = time.perf_counter() - t0
        return RunResult(
            query=query,
            answer=answer,
            label=label,
            path=path,
            trace=trace,
            metrics=self.meter.summarize(total),
            subtasks=subtasks or [],
        )
