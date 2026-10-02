# dot-agent

本地小模型与云端大模型协同的高效智能体系统 —— **路由框架骨架**（大创项目：中山大学网络空间安全学院本科生专业实践项目）。

当前阶段：**模型 API 未接入**（`gateway/stub.py` 桩客户端替身），先打通调度与路由全链路。接入真实 Ollama / 云端 OpenAI 兼容接口时，仅需实现 `ModelClient` 子类替换桩，调度层零改动。

## 快速开始

```bash
python demo/run_demo.py        # 三类典型请求的路由全流程演示
python tests/test_l1_rules.py  # L1 规则分类单测（可直接执行，无需 pytest）
python tests/test_scheduler.py # 依赖图分层 / 环退化 / 升级上限单测
# 或: pip install -e .[dev] && pytest tests/
```

## 架构与代码地图

```
请求 ─► 语义缓存 ─► 分类(L1规则→L2语义) ─► 快速通道 or 重流水线
                                            │
              ┌─────────────────────────────┘
              ▼
        分解(decomposer) → 依赖图(scheduler) → 逐层并行执行
              每子任务: 信号收集(signals) → 路由分配(router)
                        → 生成 → 校验(verify) → 失败升级(escalator, ≤K)
              → 本地汇总 → 写缓存 → 三指标(meter)
```

| 模块 | 文件 | 说明 |
|---|---|---|
| 核心类型 | `dot_agent/types.py` | 标签/特征/子任务/信号/计量记录 |
| 模型网关 | `dot_agent/gateway/` | `base.py` 抽象接口、`stub.py` 桩、`metered.py` 计量装饰、`meter.py` 三指标 |
| L1 规则层 | `dot_agent/classifier/l1_rules.py` | 零成本特征提取 + 三态判定；词表在 `config/classify_rules.json` |
| L2 语义层 | `dot_agent/classifier/l2_semantic.py` | SLM few-shot 分类（max_tokens=4） |
| 分类编排 | `dot_agent/classifier/pipeline.py` | 采纳阈值(0.85)与快速通道阈值(0.95)分离 |
| 分解器 | `dot_agent/orchestrator/decomposer.py` | 编号子任务解析，失败重试→整题升级 |
| 依赖图 | `dot_agent/orchestrator/scheduler.py` | 拓扑分层、同层并行、环→线性链、只注入直接前驱 |
| 信号收集 | `dot_agent/orchestrator/signals.py` | α-quantile + k=3 一致性；greedy 结果复用 |
| 路由 | `dot_agent/orchestrator/router.py` | 类型→策略表 POLICY + 阈值分配 |
| 升级控制 | `dot_agent/orchestrator/escalator.py` | 串行升级 ≤K，超限整题收敛云端 |
| 校验 | `dot_agent/orchestrator/verify.py` | 四种校验骨架（一致性/答案等价/约束/执行） |
| 语义缓存 | `dot_agent/cache/semantic.py` | exact 版；embedding 检索占位 |
| 总编排 | `dot_agent/pipeline.py` | `AgentPipeline.run()` 端到端，含 L3 重分类钩子 |

设计文档见桌面《任务分类技术实现路线.md》（分类三层架构/策略映射/评估协议）。

## 三指标口径（对齐申请书）

- **Acc**：外部评测器按固定评分规则判定（未实现，`eval/` 待建）；
- **C_API**：仅云端调用 in+out token 总和，含规划/校验/重试全部路径（`metered.py` 网关层统一记录，无漏记）；
- **C_time**：请求到结果的墙钟时间，本地推理计入，报告中位数/P95（`meter.summarize()`）。

## 接入真实模型（下一步）

```python
# 实现 OllamaClient(ModelClient)：POST /api/generate，验证能否返回 logprobs；
# 不能则 token_probs 传 None，α-quantile 自动退化为一致性信号（signals.py 已兼容）。
# 实现 CloudClient(ModelClient)：任意 OpenAI 兼容接口（DeepSeek/GLM/Qwen）。
pipeline = AgentPipeline(OllamaClient(...), CloudClient(...))
```

## 路线图

- [x] 路由框架骨架（本仓库）：分类三层 / 分解 / 依赖图 / 信号路由 / 升级控制 / 缓存 / 三指标
- [ ] 接入 Ollama 量化小模型（Qwen3-4B/8B Q4）与云端 API
- [ ] v0 三基线：纯本地 / 纯云端 / 规则协同可切换
- [ ] 真实校验器：答案等价（SymPy）、约束谓词库、工具沙箱
- [ ] 评测框架 `eval/`：GSM8K / MMLU 子集 / 自建中文任务，四组对照 + 消融
- [ ] embedding 相似缓存、α-quantile 阈值扫描、（可选）α-Tree + Adapter
