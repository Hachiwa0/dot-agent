# dot-agent

本地小模型与云端大模型协同的高效智能体系统（大创项目：中山大学网络空间安全学院本科生专业实践项目）。

**当前状态**：路由框架完整可运行——三层任务分类、任务分解、依赖图调度、信号路由、工具沙箱、语义缓存、三指标计量、Web 演示面板、评测框架全部就绪；模型后端默认桩客户端（零模型可完整演示），接 Ollama + 云端 API 即为真实系统。

## 快速开始

```bash
# 1. Web 演示面板（推荐入口，零模型即可演示完整路由流程）
python -m dot_agent.server          # 打开 http://localhost:8765

# 2. 命令行演示
python demo/run_demo.py

# 3. 测试（无需 pytest，直接执行）
python tests/test_l1_rules.py && python tests/test_scheduler.py && python tests/test_tools_verify.py

# 4. 评测（stub 烟测 → 四模式对照报告）
python eval/runner.py --all         # 输出 eval/report.md

# 5. 接真实模型
export OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=qwen3:4b   # 可选，自动探测
export CLOUD_BASE_URL=https://api.deepseek.com/v1 CLOUD_API_KEY=sk-... CLOUD_MODEL=deepseek-chat
python -m dot_agent.server         # 之后同一命令，自动切换真实端云
```

## 架构

```
请求 ─► 语义缓存 ─► 分类(L1规则→L2语义) ─► 快速通道 or 重流水线
                                            │
              ┌─────────────────────────────┘
              ▼
        分解(decomposer) → 依赖图(scheduler) → 逐层并行执行
              每子任务: 工具沙箱(TOOLUSE) 或 信号收集(α-quantile+k=3一致性)
                        → 路由分配(router) → 生成 → 校验(verify) → 失败升级(escalator,≤K)
              → 本地汇总 → 写缓存 → 三指标(meter)
```

| 模块 | 文件 | 说明 |
|---|---|---|
| 核心类型 | `dot_agent/types.py` | 标签/特征/子任务/信号/计量记录 |
| 模型网关 | `dot_agent/gateway/` | `base` 抽象 · `stub` 桩 · `demo_behavior` 演示行为 · `ollama` 本地真实 · `openai_compat` 云端真实 · `metered` 计量装饰 · `factory` 自动装配 |
| 分类 | `dot_agent/classifier/` | L1 规则（词表外置 `config/classify_rules.json`）· L2 SLM few-shot · 编排（采纳/快速通道双阈值） |
| 编排 | `dot_agent/orchestrator/` | 分解 · 依赖图（环退化/同层并行/前驱注入）· 信号（α-quantile+一致性，greedy 复用）· 路由策略表 · 升级守卫(≤K) · 校验器 · 工具沙箱 |
| 缓存 | `dot_agent/cache/semantic.py` | exact 版 + 确定性域过滤；embedding 检索占位 |
| 总编排 | `dot_agent/pipeline.py` | 端到端 + L3 重分类钩子 + `use_signals` 开关（v0/v1 对照） |
| 服务端 | `dot_agent/server.py` + `web/` | 标准库 HTTP；面板：三指标卡/依赖图 SVG/决策时间线/历史 |
| 评测 | `eval/` | 四组内置数据集 · 四模式 runner · 报告 |

## 三指标口径（对齐申请书）

- **Acc**：固定评分规则（数值容差/选项字母/约束谓词/内容包含）；
- **C_API**：仅云端 in+out token，含规划/校验/重试全部路径（`MeteredClient` 唯一出口，内存 + SQLite 双落库）；**规划开销**（分类/分解/依赖判断/汇总）单独统计——DoT 论文未计量此开销，是本项目报告的差异点；usage 缺失记 `unknown_usage_calls`，不并入也不估算；
- **C_time**：请求到结果墙钟时间（本地推理计入），报告中位数/P95，附阶段拆解；**请求级量纲**（SQLite tasks 表按请求落库，分位数不失真）；真实模型部署必须先 warmup 预热；
- **降级率**：云端不可用降级本地时如实记双档位（`cloud_fallbacks`），不虚报云端消耗。

所有调用记录落 `data/metrics.db`（calls/tasks 双表，run_id 分组），评测报告引用原始库。

## 团队整合

本项目合成了团队两份实现的长处（计量工程 ← cloud-edge-agent；部署与诚实性纪律 ← edge-cloud-prototype），完整分析、吸收清单与未采纳理由见 **[docs/TEAM_INTEGRATION.md](docs/TEAM_INTEGRATION.md)**；版本固定部署纪律见 **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**。

## 评测四模式

`local_only`（整题本地）/ `cloud_only`（整题云端，上界参照）/ `rule`（v0 规则协同，信号关闭）/ `signal`（v1 信号协同）——对应申请书"纯本地、纯云端、规则协同、优化协同"四组对照；消融按模式差异与 `use_signals` 开关展开。

## 路线图

- [x] 路由框架：分类三层 / 分解 / 依赖图 / 信号路由 / 升级控制 / 缓存 / 三指标
- [x] 工具沙箱（mock 天气/提醒/计算器 + 错误码分类）与真实校验器
- [x] Web 演示面板（路由决策可视化）
- [x] 评测框架与四模式对照
- [ ] 真实模型接入验证（Ollama logprobs 可行性 → 决定 α-quantile 或一致性路线）
- [ ] embedding 语义缓存、阈值扫描、（可选）α-Tree + Adapter
- [ ] 隐私扩展（申请书后期阶段）
