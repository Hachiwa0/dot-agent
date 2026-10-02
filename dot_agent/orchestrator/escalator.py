"""升级控制：串行升级次数上限 K，超限整题收敛为一次云端调用。

对应申请书"限制串行升级次数"与拟解决问题（1）"避免频繁误升级或
多次串行调用"——所有升级路径必须经过本守卫，且 retry token 全额计量。
"""
from __future__ import annotations

DEFAULT_MAX_SERIAL = 3


class EscalationGuard:
    def __init__(self, max_serial: int = DEFAULT_MAX_SERIAL) -> None:
        self.max_serial = max_serial
        self.count = 0
        self.converged = False  # True → 后续不再逐子任务升级，整题交云端

    @property
    def remaining(self) -> int:
        return max(0, self.max_serial - self.count)

    def can_escalate(self) -> bool:
        return not self.converged and self.count < self.max_serial

    def escalate(self) -> bool:
        """记录一次升级；达到上限自动置收敛标志。返回是否仍允许继续。"""
        if not self.can_escalate():
            return False
        self.count += 1
        if self.count >= self.max_serial:
            self.converged = True
        return True

    def converge_to_cloud(self) -> None:
        """外部触发整题收敛（如分解失败、重分类超限）。"""
        self.converged = True
