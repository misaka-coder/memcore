# 第一轮研究：先把一个问题测清楚

**实验定位更正（2026-09-06）：首轮是受限宿主预实验，尚不是 Akane 实际运行链的正式对照。**六次工具上限和直接拒绝格式错误是实验执行器行为，不能据此把5/12算作 Akane 的成功率。下一步先对齐真实工具循环、最终模板和恢复链，详见 [宿主一致性更正](host_fidelity_correction.md)。

更新：2026-09-06。首轮四组真实实验和十二个独立探针已完成：八项基础任务的核心内容达到标准（其中一项额外展开越出要求范围），五个探针交付合格终答，另有三项 JSON 格式失败、四项工具调用超限。含先前预检共82次请求，时段费率估算0.1665442元。见 [首轮报告](first_real_experiment_report.md)、[结果数据](first_real_results.json)；全部失败与原始轨迹均保留。

本轮已将一个设计问题变成可检查的真实实验。接下来的研究方向由成功证据和失败原因决定，再扩展对照并组织论文；本轮没有同时验证整个 Akane + MemCore。

完整设计见 [设计动机与研究问题](../../akane_memcore_design_rationale_and_research_v1.md)。本包是其中实验计划的前置小试，先做两个条件；原文的缓存布局 × 结果保留策略实验留到下一阶段。

## 1. 这次只问一个问题

**完成工具任务后，把长结果变成可回读卡片，能否减少后续请求中的上下文占用，同时让模型在追问时正确找回旧版本细节？**

| 条件 | 开放工具轮 | 最终回复提交后 |
| --- | --- | --- |
| full：对照组 | 展示完整结果 | 继续展示完整结果 |
| card：实验组 | 展示完整结果 | 按实际策略生成卡片，允许按来源 ID 回读 |

两组使用相同模型、人格、工具说明和脚本。这里只改变 `operation_projection_policy`。短结果可能不生成卡片，所以必须检查实际处理记录，不能只看配置开关。

这里的“对照组”就是拿来比较的另一种设置；“探针”就是任务之后专门追问历史细节的问题；以后提到的“消融实验”，就是去掉某个部件，看看它实际贡献了什么。

## 2. 养成动态提示词如何处理

第一轮两组均关闭养成功能，保留相同的固定人格。底部动态信息是否影响行为，与它放在提示词哪个位置是两回事：Akane 的养成规则会引导饥饿时讨食、精力低时减少说话或休息，因此会改变答题行为，而不仅是增加 token。

本轮实际复用 Akane `PromptBuilder` 与 `care_enabled=False` 的 `desktop_pet` profile，32次 builder 调用验证16个步骤的稳定人格与提示，移除 `care_runtime`、`state_request`，无实时养成/视觉/资源输入。完整 Engine 未启动，因此只称为宿主提示契约验证。源码中的完整实验实例还可经 `InstanceContext.features.care=False` 停用养成模块，相关路径见 [接入记录](akane_adapter_preflight.md)。

这些是独立实验实例的设置，本次没有修改日常 Akane 配置。保留测试脚本中的时间锚点；按脚本推进日期，不让机器实际运行耗时决定角色状态。

后续如果研究“完整陪伴系统中是否仍然有效”，再给两组回放**同一段预先记录的养成状态序列**。这样既保留真实产品因素，又避免一组调用更慢、自然衰减更多，导致两组状态不一致。关闭养成的结果不能直接证明真实陪伴体验更好。

## 3. 已经准备好的材料

| 文件 | 用途 |
| --- | --- |
| [scenarios.json](scenarios.json) | 两个虚构场景、版本化工具资料、逐步用户消息 |
| [answer_key.json](answer_key.json) | 仅供评分者使用的答案、来源与评分标准 |
| [run_manifest.json](run_manifest.json) | 两种配置、四个运行条目、预算及待记录指标 |

场景一是“闲聊约定 → 查询场地资料 → 资料更新 → 闲聊 → 追问旧资料”；场景二是“表达偏好 → 阅读虚构代码 → 代码更新 → 闲聊 → 追问旧参数”。每个场景各有三个探针：旧工具细节、聊天事实、资料明确未提供的字段。

两场景 × 两条件，共四条基础运行。每条运行在闲聊结束后保存快照，三个探针各自从这个快照开始，避免前一道题的回答给后一道题提示。这不是四次 API 请求；每段对话、工具续接和探针都可能产生调用。

这些 JSON 是实验规范，不能直接作为现有 `memcore.eval` 的数据集运行。现有离线检索评估使用模拟 LLM 与哈希向量，可检查部分检索流程，不能替代真实聊天模型自主选择工具的实验。

## 4. 模型与预算

按作者选择，使用官方 `deepseek-v4-flash`。作者给出的 50–100 暂按人民币理解；首阶段总支出上限为 **50 元，包含所有条件、探针、重试和预检中的收费请求**，不要求花完。

截至本次核对，官方文档支持该模型的工具调用、JSON 输出和非思考模式。首轮拟显式设置 `thinking.type=disabled`，两组一致，以简化协议验证；这不是对思考模式效果的判断。温度设为零也不保证完全确定性。[模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[思考模式协议](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)。

正式运行时记录实际返回的模型信息、请求配置及当时价格。执行器必须在发请求前，按剩余预算和该请求最坏费用决定是否继续；预留重试费用，缺少用量字段记为 `null`，不要记成零。凭据通过安全的本地配置绑定，不写进实验材料或对话。

## 5. 当前完成状态与后续方向

1. **真实适配和首轮已完成。**实验宿主通过公开 `MemorySystem` 生命周期，使用本地 BGE-M3、真实 Akane 固定人格提示及官方模型的自主原生工具循环；完整 Akane Engine 验收仍是另一项工作。
2. **先恢复真实宿主行为。**复用 Akane 已有的工具循环、最终模板、schema 与验证/恢复，移除实验新加的六次上限。先完成宿主等价验收，再考虑空值说明等单变量消融；保留首轮结果。
3. **答案与证据分开核对。**五个合格终答中只有一个包含实际旧正文回读；正文可见时直接答对不等于完成了回读。
4. **再扩大研究。**增加场景、长度和重复次数，加入删结果、普通摘要等对照，并独立验证时间导航、三层记忆或缓存布局。根据证据确定论文范围。

`scenarios.json` 与 `run_manifest.json` 继续作为冻结前的准备规范，保留 `prepared_not_run` / `not_run` 模板，供校验与未来新实验使用；当前执行结果单列在 [first_real_results.json](first_real_results.json)，不把已执行数据混回输入规范。收费阶段必须继续复用原账本。

### 已可运行的离线命令

在仓库根目录执行：

```text
uv run --extra dev python -m examples.research_pilot validate
uv run --extra dev python -m examples.research_pilot smoke
```

代码入口见 [examples/research_pilot](../../../examples/research_pilot/__main__.py)。默认产物进入 Git 忽略的 `.research-runs/` 下新建目录，输出位置会打印在终端；可用 `--output` 指定一个尚不存在的目录。已有目录会被拒绝，不覆盖历史记录。

每次生成 `report.md`、`report.json`、逐事件 `events.jsonl`，以及各基础运行的 journal、来源映射和独立实验数据库。模型请求、脚本响应、工具结果、开放轮正文检查、终局投影指标和失败都可核对。工具 observation 只在 `payload.output` 保存一份正文；回读时解开渲染包装，比较其中 output 与初次返回的正文，避免重复或误把包装差异当正文丢失。

当前 MemCore 没有公开的快照导入/导出门面，因此探针分支通过公开 API 重放同条件已经记录的 journal，保留相同 source/call IDs，在独立硬命名空间、数据库和索引中恢复历史。恢复后的 provider 历史与来源映射必须和基础运行一致，不重新询问模型，也不把重放计成新增模型请求。checkpoint 用于这些探针分支的恢复；当前 CLI 尚不提供崩溃后自动续跑收费实验。

离线驱动只支持单工具调用的模拟轮次，显式禁止网络；使用 HashedEmbeddingProvider 仅检查读写装配。报告标记 offline_smoke、forced_routing=true、实际模型为脚本驱动，真实 provider usage 与答题分数为 null，收费为零。validate/smoke 不读取 API 凭据，也不调用独立收费 preflight 路径。此处的“通过”只指本地机制与执行流程。

离线 smoke 的入参准入使用带标签的 UTF-8 估算，供应商 token 为空。真实 live 已注入经健康验证的生产 BGE-M3，并同时保留估算准入和供应商实际 usage。首轮从新建源码副本运行，逐文件校验哈希；它仍未完全锁定依赖和远端服务版本。

离线 smoke 不加载 Akane；真实 live 使用关闭养成的真实 builder 提示包，保留其分块布局和脚本时间，并校验当前刺激只出现一次。完整 Engine 和缓存布局的因果效果仍未验证。

### 独立收费传输预检

实现见 [preflight.py](../../../examples/research_pilot/preflight.py) 与 [budget.py](../../../examples/research_pilot/budget.py)。它从进程环境读取 DEEPSEEK_API_KEY，只向固定官方 HTTPS endpoint 发送合成内容，不读取 Akane 实例配置。先强制调用一次 pilot_echo，再携带完整支持的 assistant 消息、原始调用 ID 和参数字符串回传结果，最后验证 MemCore 最终 JSON 解析。报告保留 forced_routing=true、memory_retrieval_verified=false、akane_runtime_verified=false。

先在原仓库执行以下离线准备；源码目录必须是新目录：

~~~text
uv run --extra dev python -m examples.research_pilot init-budget
uv run --extra dev python -m examples.research_pilot freeze --output .research-runs/provider-preflight-source
~~~

init-budget 初始化或读取固定的 .research-runs/stage1-budget.sqlite3，不清空旧金额。freeze 只复制源码与合成 JSON，绑定既有账本身份和相对位置。随后在冻结源码目录中，用原仓库的同一 Python 环境执行：

~~~text
python -m examples.research_pilot preflight --output ../provider-preflight-result
~~~

**只有最后一条命令会收费。**它要求当天已人工核对官方费率，并校验源码哈希和既有账本身份。账本丢失、身份不符或有未决费用都会停止。整个阶段必须复用该账本，不删除、重建或复制出独立收费分支。源码冻结尚未锁定 Python/第三方依赖，不称为完整可复现环境。

派发前按官方 1M 上下文上界、2048 输出与峰值费率预留 3.16416 元。收到同一模型的有效 usage 后释放未使用预留；超时、错误、异模型或缺少有效用量都保留预留并停止，不自动重试。按请求时段费率计算的费用另列为估算，供应商账单金额保持 null。请求捕获不记录 Authorization、密钥或错误响应正文。

本预检不使用 embedding。后续完整宿主隔离与入口见 [Akane 接入检查](akane_adapter_preflight.md)。

### 真实场景执行器

入口为 [live.py](../../../examples/research_pilot/live.py)。先用 [akane_prompt.py](../../../examples/research_pilot/akane_prompt.py) 在隔离环境中导出真实提示包，并用 [production_embedding.py](../../../examples/research_pilot/production_embedding.py) 验证现有本地生产模型。随后使用已有阶段账本冻结源码和这两份输入：

~~~text
python -m examples.research_pilot freeze --output .research-runs/new-live-source --prompt-bundle <verified-prompt-bundle> --embedding-report <verified-embedding-report>
~~~

从该冻结源码目录，使用具备本地模型依赖的同一 Python 环境运行 `python -m examples.research_pilot live --output ../new-live-result`。这条命令会收费；凭据仍由进程环境绑定，模型本地位置经 `PILOT_EMBEDDING_MODEL_PATH` 传入且不进入提示或公开报告。目录必须为新目录，原账本不能重置或复制；当天费率、源码/输入哈希、embedding 健康、模型信息和每次最坏费用准入都会检查。实际实现不自动重试。

真实版支持 provider 批量原生工具、完整参数字符串及结果边界。它保存逐请求/逐轮 trace、独立 checkpoint、事件与失败；评分由离线评估器提取候选，再逐项语义审阅。首轮已经运行，不需要为查看结果重跑收费命令。

## 6. 执行器必须兑现的实验约束

- 模型只能看到当前步骤的用户消息与实际选择的工具结果，不能看到未来步骤、评分规则或答案文件。工具采用允许列表：本场景的 `lookup_fixture` 与指定记忆读取工具；不提供文件、shell、网络或通用工具发现入口。
- 每个条件、场景及探针分支隔离存储和索引，按 manifest 隔离硬命名空间并保持来源关系。快照包含对应的历史、provider 消息与宿主状态，不能只换 `conversation_id`。记录逻辑资料 ID 到真实 MemCore 来源 ID/hash 的映射。
- 资料服务更新后，两组都不能重新取到 v1；请求旧版本时返回明确的不可用状态。历史正文仍可通过 MemCore 读取，不能把重做原工具查询误计为记忆回读。
- 两组开放任务轮必须实际收到完整资料，并成功提交 final。card 组记录实际 settlement/noop/failure 及前后 token；处理失败的运行留下失败标记，不能算有效比较。
- 审计探针前所有模型生成且后续可见的输出，包括新版比较、闲聊和协议要求回传的内容。提前说出答案的运行保留并标记；答对不能自动算回读成功。探针正确性与工具证据分别评分。
- Phase 0 不调用 raw/semantic maintenance，以隔离工具结果投影策略。因此聊天事实探针只检查连续性，不证明长期记忆恢复；“资料明确缺字段”也只检查不编造该字段，不能泛化为一般拒答能力。
- 输入用量包括工具循环、回读及重试。账本按实际 API 请求唯一计费，共享的前置步骤不因三个探针分支重复计费。达到上下文、调用次数或预算上限时停止并记录原因。
- 记录实际执行顺序与间隔。供应商缓存可能被先跑的条件预热，数据库隔离不能隔离供应商缓存。本轮费用只用于预算和链路记录，不能由 full/card 的单次账单差直接推断策略节费；费用研究另做交错/反向顺序和重复实验。

## 7. 首轮结果怎样才算有用

四条运行都具有完整成功或失败记录，能对应到实际配置、来源和评分；收费账本可核对；没有隐藏删去的失败或答案泄露。满足这些条件，即使 card 没有更好，这轮也有研究价值：它能区分回读机制、模型工具选择、提示或实验设计中的问题。

最初只报告逐例结果和调用轨迹。两个手工场景、每条件一次，不能用于总体显著性、通用 Agent 优越性、长期陪伴体验或新颖性结论。后续需要更丰富的任务、重复运行、合理对照和公开可复现材料。

## 8. 第一批只精读三份资料

| 资料 | 先带着哪个问题阅读 |
| --- | --- |
| [MemGPT](https://arxiv.org/abs/2310.08560) | 它怎样让模型管理分层上下文？与我们的控制边界、回读方式有什么相同和不同？ |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | 历史记忆问题怎样分类、构造答案和评分？哪些评价方式适合我们，哪些不能直接迁移？ |
| [Anthropic Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) | 已有工具结果清理如何处理调用参数和结果？我们的可回读卡片还需证明什么额外价值？ |

每篇先记四项：解决的问题、机制、实验、局限。“保留调用、移除旧结果”已有相关实践，不能单独宣布首次提出；组合机制是否构成研究贡献，需要更广的文献对照及实验支持。读完这三份，再沿最相关的引用扩展。
