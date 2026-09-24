# 首轮实验与 Akane 实际宿主不等价：结论更正

日期：2026-09-06。作者指出 Akane 不设固定工具上限，并已有严格输出模板。重新对照当前源码与冻结实验后，确认首轮没有充分保持宿主行为一致。

**若目标是评价 Akane 当前真实运行链中的 MemCore 策略，这轮实验不足以回答。**应将它归类为“复用 Akane 提示的受限实验宿主预实验”。5/12 是该执行器及其人为边界下的合格终答数，不是 Akane 的成功率。先前将这些失败概括为 Akane 或其实际调用链的问题，归因范围过大。

## 已确认的差异

| 项目 | Akane 当前源码 | 本轮冻结执行器 | 影响 |
| --- | --- | --- | --- |
| 工具上限 | `TOOL_ROUND_HARD_LIMIT` 默认0；Engine 在0时继续循环 | 固定最多6个工具调用，第7个不执行且直接失败 | 四个超限结果只能说明实验被此上限截断；Akane 最终能否完成尚未观察 |
| 计数和停止 | 配置为有限上限时按实际执行的工具批次计数；无效模型决策另有修复计数；达到有限上限还会生成收尾回复 | 按调用个数累计；包括返回参数错误的工具；超限直接终止 | 即使数字设成相同，行为也不等价 |
| 最终输出 | 原始模式模板 + LLMRuntime 解析 + Engine 终答验证及同上下文修复 | 将桌宠输出契约覆盖为仅 speech/memory_metadata；直接用 MemCore 严格解析，失败即终止 | 三个格式失败不能当成 Akane 经真实恢复流程后的终态 |
| 工具 schema | 先取 MemCore plain 规格，经 capcore provider 默认 `strict=False` 构建，保留原可选参数 | 直接调用 MemCore 原生 strict schema，所有可选字段改为必填可空 | 改变了模型填写空值与时间参数的负担，不能将本轮参数错误率外推给 Akane |
| 宿主入口 | Engine 的完整轮次与续接、验证、恢复链 | 只实际复用 PromptBuilder；工具循环与提交由实验代码另写 | 提示来源真实，不等于宿主行为等价 |

源码定位：Akane `config.py:307`、`tool_orchestration_engine.py:94`、`engine.py:4495` 和 `engine.py:4674`；最终生成入口和重试见 `engine.py` 的 `_stream_final_response`，JSON 提取见 `llm_runtime.py:3654`。实验对应 `examples/research_pilot/live.py:198`、`:225`、`:332`，以及 `akane_prompt.py:47`。本次只读源码，没有加载日常配置或访问运行数据，因此代码默认值的核验与作者对实际配置的说明分别保留。

schema 路径另经实际安装包核对：Akane `capability_registry.py:65` 使用 `tool_format="plain"`，`native_tool_schema.py:74` 未覆盖 provider 的 strict 默认值；已安装 capcore-provider-openai 的构建器默认 `strict=False`，直接保留输入 schema。实验 `runner.py:179` 使用 MemCore 的 OpenAI schema 默认 `strict=True`。这不是仅换了函数名字，两条模型可见 schema 的 optional/required 语义不同。

## 对 JSON 失败的补充核验

离线抽取 Akane 当前源码中的纯解析方法，送入原先三份被判 `invalid_final_json` 的实际原始响应，三份都能找到内部 JSON 对象及 speech。这项检查没有启动宿主、没有新增模型请求。诊断见 [解析核验](../../../.research-runs/pilot-live-20260906-result/analysis/host-fidelity-parser-check.json)。

这不等于三项在真实 Engine 中一定成功：Engine 还会验证 JSON 外文本、内容和终答状态，并可能重试。它证明的是“实验的直接拒绝”与“Akane 的完整解析和恢复链”不能混为同一个判定。原三项失败记录保留，不事后改成成功。

当前 Akane 主聊天和原生工具路径也没有强制 `response_format=json_object`；因此不能简单把问题解释为“实验忘开 JSON mode”。应当核对的是完整输出模板、请求构建、解析、验证和恢复流程。模板可以降低格式错误概率，但模板存在本身不是无错误的数学保证。

## 为什么两组都设六次仍不能代表 Akane

full 可以直接使用已可见正文，card 通常要额外回读；它们所需调用和错误恢复步数可能不同。同一个人为上限会与被比较的策略相互影响。

因此首轮可以测“六次调用预算下，这个实验宿主能否完成”，却不能替代“Akane 不设固定轮数上限时最终能否完成”。外部费用止损若中止实验，应记录为预算/实验中止及未观察到最终结果，不能自动解释为宿主的功能失败。无限轮数也不意味着必然成功，现有四条超限轨迹无法推断后续结局。

## 仍可保留的证据

- 本地生产 embedding 已加载并通过健康检查。
- 关闭养成的真实 Akane PromptBuilder 提示确实被使用。
- 四份长资料的终局卡片处理实际发生，正文仍可完整回读；模型也有一次成功回读。
- 原始请求、结果、失败和费用计账是真实可核对的。

这些支持接线和局部机制验证。它们不支持把5/12当作 Akane 正式分数，也不支持归因于 MemCore 导致 Akane 可靠性下降或上升。

## 正式对照应先恢复什么

优先复用 Akane 的实际请求构建、工具规格、工具续接、最终模板、验证/恢复和 MemCore 提交链，在隔离实验环境中关闭养成，只改变被研究的 `operation_projection_policy`。先核验真实请求与轮次行为的等价性，再重复四组场景。

本轮新加的六次工具上限应从正式宿主行为中移除；阶段50元的外部预算约束继续复用原账本。格式和工具错误后的恢复使用 Akane 已有策略，所有恢复请求仍计入费用。不能先给 Akane 添新提示或修改它的工具 schema，再称为对现有产品的忠实验证。

此次更正没有改写冻结源码、原始请求、原始评分或失败轨迹，也没有新增收费调用。后续正式结果必须另存为新实验，不能覆盖这一轮。
