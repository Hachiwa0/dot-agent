"""调度器与升级守卫单测：拓扑分层、环退化线性链、K 次上限。"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.gateway.stub import StubModelClient  # noqa: E402
from dot_agent.orchestrator.escalator import EscalationGuard  # noqa: E402
from dot_agent.orchestrator.scheduler import TaskScheduler  # noqa: E402
from dot_agent.types import ModelTier, SubTask  # noqa: E402


def _subs(*descs: str) -> list[SubTask]:
    return [SubTask(id=f"s{i}", description=d) for i, d in enumerate(descs, 1)]


def test_levels_parallel_same_batch():
    subs = _subs("a", "b", "c", "d")
    edges = [("s1", "s3"), ("s2", "s3"), ("s3", "s4")]
    levels = TaskScheduler.build_levels(subs, edges)
    assert [[st.id for st in lv] for lv in levels] == [["s1", "s2"], ["s3"], ["s4"]]
    assert subs[0].deps == [] and subs[2].deps == ["s1", "s2"]


def test_cycle_falls_back_to_chain():
    subs = _subs("a", "b", "c")
    edges = [("s1", "s2"), ("s2", "s3"), ("s3", "s1")]  # 成环
    levels = TaskScheduler.build_levels(subs, edges)
    assert [len(lv) for lv in levels] == [1, 1, 1]  # 线性链


def test_no_edges_linear():
    subs = _subs("a", "b")
    levels = TaskScheduler.build_levels(subs, None)
    assert [len(lv) for lv in levels] == [1, 1]


async def _run_with_edges(edges_text: str):
    async def dep_behavior(prompt, caller):
        return edges_text

    client = StubModelClient(ModelTier.MD, behavior=dep_behavior)
    scheduler = TaskScheduler(client)
    order: list[str] = []

    async def executor(st: SubTask, ctx: dict[str, str]) -> str:
        order.append(st.id)
        assert all(d in ctx for d in st.deps)  # 前驱必须已完成
        return f"ans-{st.id}"

    subs = _subs("a", "b", "c")
    answers = await scheduler.run("q", subs, executor)
    return answers, order, subs


def test_scheduler_run_order_and_ctx():
    answers, order, subs = asyncio.run(_run_with_edges("s1 -> s3\ns2 -> s3"))
    assert set(order[:2]) == {"s1", "s2"} and order[2] == "s3"  # 同层并行、跨层串行
    assert answers == {"s1": "ans-s1", "s2": "ans-s2", "s3": "ans-s3"}


def test_escalation_guard_limit():
    g = EscalationGuard(max_serial=3)
    assert g.can_escalate()
    assert g.escalate() and g.escalate() and g.escalate()
    assert g.converged and not g.can_escalate() and not g.escalate()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            if inspect.iscoroutinefunction(fn):
                asyncio.run(fn())
            else:
                fn()
            print(f"PASS {name}")
    print("all passed")
