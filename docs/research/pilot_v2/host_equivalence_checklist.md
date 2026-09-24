# 第二轮宿主等价验收

状态：这是收费重测前的验收规范，不是已通过的证明。验收用隔离的新宿主状态与脚本化 SDK 响应；脚本化结果只证明接线，不进入正式模型得分。不能用第一轮“真实 PromptBuilder 导出成功”代替 Engine 回合验收。

## 1. 等价范围与源码依据

目标是保留当前 Akane 的真实文本回合主线，仅接入受预算管理的传输、生产 embedding、指定合成资料工具与隔离存储。客户端使用真实 `desktop_pet` 提示配置，关闭养成；不声称覆盖桌面显示、TTS 或完整产品交互。

下表的文件名以相应仓库根目录为起点。正式验收应记录实际加载文件的哈希和包版本，不能只检查类名。

| 权威实现 | 已核对的行为 | 对新宿主的要求 |
| --- | --- | --- |
| Akane `companion_v01/engine.py`：`AkaneMemoryEngine.__init__`、`process_turn`、`_run_turn_core` | 真实构造同时建立宿主 MemoryStore、LLMRuntime、MemcoreManager、提示配置、工具注册与生命周期；`process_turn` 有开放 MemCore 回合的退出保护 | 调用真实构造及 `process_turn`；不能用 `__new__` 加若干同名方法、MockEngine 或复制回合循环声称完整等价 |
| Akane `companion_v01/engine.py`：`_prepare_final_response_context`、`_build_final_response`、`_build_final_response_request_kwargs`、`_final_attempt_terminal_output` | 每次模型结果经过真实解析、归一化、工具判定或最终回复恢复 | 不能用宿主自己的 `parse_chat_output` 成功/失败判断替代这一主线 |
| Akane `companion_v01/llm_runtime.py`：`call_chat_json_result`、`call_chat_text`、SDK 请求与 observer | 运行时形成实际 provider 消息、调用 request observer，并解析 SDK 响应 | 传输接入放在 SDK 客户端边界；保留运行时解析、原生工具映射和 projection observer |
| Akane `companion_v01/native_tool_schema.py`，capcore-provider-openai `capcore_provider_openai/chat.py` | 从真实 handler 的 canonical ToolSpec 构造 schema；当前调用使用 provider 包默认 `strict=False`，保留原有 `required` | 不复用第一轮将所有 optional 字段变为 required+nullable 的 schema |
| Akane `companion_v01/tool_orchestration_engine.py`：`max_tool_rounds`，Engine `_run_turn_core` | `TOOL_ROUND_HARD_LIMIT=0` 表示不限轮；正限额按实际执行的工具批次计数；被拒绝的模型 decision 有独立恢复次数 | 不保留第一轮六次调用截断；不得以总工具数、失败参数数替代真实 batch 计数 |
| Akane `companion_v01/prompt_profiles.py`、`prompt_builder.py`、`final_output_engine.py` | desktop_pet 常规模板包括 `emotion`、`speech`；其它字段按真实配置与用途追加，MemCore 元数据另有真实指令；归一化可添加宿主派生字段 | 保留真实原模板及归一化结果，不后置“两字段唯一”研究模板，不把宿主默认值记成模型原始输出 |
| Akane `companion_v01/memcore_integration/manager.py`：`record_tool_batch`、`build_open_turn_projection`、`record_request_projection`、`complete_input_turn` | 同轮工具正文、provider 投影及最终原文通过 MemorySystem 生命周期保存；终答提交调用 `MemorySystem.complete_turn` | 不手工写 store 私有表，不在研究 runner 另建第二套提交逻辑 |

真实 Akane handler 的记忆工具名称为 `retrieve_memory`、`read_memory_timeline`、`browse_memory`、`open_memory`。前两个在内部转到 MemCore 相应公共操作。记录与调用必须使用实际生成的名称；结果分析可分组映射，不能把第一轮 `retrieve_for_turn` / `read_timeline` schema 偷换进新宿主。

允许替换的依赖是收费 SDK 传输、无网络的脚本 SDK 响应、embedding 构造端口和合成资料 handler 注册。可以隔离配置、时间来源和随机 ID 来源，但必须列出具体注入位置及效果。`process_turn`、生成/恢复函数、工具编排/归一化、MemCore 提交和 projection 方法必须保持权威实现；测试 spy 只作透明观察，不能预制返回值。注入的 embedding 必须仍走真实 adapter 与检索链，不能用一个假 `retrieve_memory` 方法代替生产向量。

## 2. 最小阻断验收

每项保留“脚本输入、实际 SDK 请求摘要、真实调用链、关键断言和结果”。H1—H7、H9—H10 是本次目标的最小阻断项；H8 是提交故障路径的补充诊断，只有接入改动影响这一语义时才追加为阻断项。失败项修复后重跑相关验收；验收记录与正式模型结果分开。

| 编号 | 最小案例 | 判定依据（oracle） |
| --- | --- | --- |
| H1 真实回合入口 | 新建完整 Engine，脚本 SDK 返回原模板 JSON；通过 `process_turn` 发送一句用户输入 | 实际构造/调用来自冻结 Akane 源码；观察到 `_run_turn_core`、LLMRuntime、MemcoreManager 的调用；实际 user 与 final 各提交一次，无开放回合残留。仅 `isinstance`、导入成功或 mock 方法被调用不算通过 |
| H2 超过六次工具 | 同一回合连续给出七次合法、各有独立 call ID 的合成工具调用，随后 SDK 响应给合法终答；按真实配置设置不限轮或大于七的批次限额 | 七次都由真实工具编排执行，七个原生 call/result 配对依次进入模型后续请求，最后回复经 Engine 提交；不存在研究 runner 的第七次截断。每个执行批次的数量可与请求轨迹对账 |
| H3 optional cursor | 先由真实资料 handler 产生可打开的来源，再原生调用 `open_memory`，只给 `memory_id` 和 `view="content"`，省略 cursor 及其他无关 optional 字段 | 实际出站 schema 的 `required` 不含 cursor，也没有宿主新增 `strict=true`；真实 canonical validation、handler、manager 调用成功。后续请求接收实际返回正文，不能仅直接调用 MemorySystem 证明这一点 |
| H4 原始最终模板 | 用相同真实 desktop_pet/care-disabled 配置构造提示，SDK 返回包含 `emotion`、`speech` 和正确 `memory_metadata` 的原模板 JSON | 没有两字段研究覆盖指令；输出通过真实解析和归一化。原始 provider 文本、规范化 speech/metadata、宿主派生 presentation 字段分别可追溯；没有因为 emotion 或真实可选字段而拒绝输出 |
| H5 JSON 外正文恢复 | 一次工具返回后，SDK 返回“外部解释段 + 合法 JSON”，下一次返回合法原模板 JSON | Engine 实际触发 `json_external_text` 恢复，而不是提取 JSON 即成功或直接失败结束。第二次请求保留相同系统前缀、工具 schema、用户刺激、已完成工具结果和 cache key；只在允许的请求尾部增加恢复反馈。坏结果不被当成已接受的最终 speech；恢复后的模型正文按真实提交路径保存 |
| H6 完整恢复尾路 | SDK 连续返回无法交付的破损 JSON，达到真实 `CHAT_MODEL_DECISION_MAX_ATTEMPTS` 后给一次完整纯文本 | 真实 `_recover_final_response_plain_text` 经 `LLMRuntime.call_chat_text` 调用传输；正文由 SDK 响应产生，宿主只补 presentation 默认值并保留恢复来源；实际每次请求均可计费记账。不能直接返回 canned fallback 冒充模型答案 |
| H7 开放/终局正文保真 | 用含明显首尾标记和中间特殊字符的长合成正文，在 full/card 两策略分别执行真实工具回合 | final 前两条件实际 provider 请求均含完整工具结果；最终提交后 full 保持原投影，card 有实际 settled/noop/fallback 结果。对实际 settled 的 card，经真实 `open_memory` 路径回读并逐字核对原始工具 output；对照最终下一请求确实收到正文。只匹配 ID、标记、答案词或配置值不算通过 |
| H8 提交故障补充诊断 | 对验收用隔离依赖注入一次可恢复提交故障，再注入持续故障 | 一次故障走真实幂等 completion 重试且不重复 final；持续失败走真实 abort/结构化 `_memcore_failure`，保留真实模型回复的失败来源，无假 completed。该项验证提交链，故障必须发生在依赖端口，不能 mock 整个 finalize 方法 |
| H9 三分支同基线 | 完成五轮 setup 后建立不可变 checkpoint，各自新建三个 probe 宿主实例；在 A 分支写入独有标记后启动 B/C | 三个初始 provider 历史（不含各自新探针）、setup source 映射、原始结果 hash、相关宿主状态与 checkpoint 完全一致；B/C 和 checkpoint 均不含 A 的标记、结果或后续 tool state；不能只比较 SQLite 文件名或 message count |
| H10 无预知/无绕行 | 故意让 setup 以外的答案/未来探针带验收 sentinel；拒绝过期 v1 fixture 查询；尝试未注册工具 | 实际 prompt、schema、当前资料 handler、所有 setup authored 输出中没有 future/gold sentinel；旧资料查询返回真实 unavailable，不重提供 v1 或换成 v2。实际 tool surface 只含批准的资料与记忆工具，不能通过通用文件/代码/网络入口读取答案 |

H5/H6 的“坏 JSON”要选定能触发真实恢复的反例。当前 Akane 会接受仅带 Markdown JSON 围栏的对象，也可能直接包装完整普通纯文本，或恢复可证实的完整 speech；这些不是应强制再请求的反例。原生工具 decision 也不是坏终答，修复请求仍可返回一个合法工具调用并回到普通工具循环。不能为了验收通过修改这些真实语义。

除了 H2 的不限轮主案例，应再用小正限额校验耗尽行为：真实 Engine 保留工具 schema，显式改为禁止继续工具的选择策略并请求收尾。它不应被研究 runner 的直接异常终止所替代。非法工具 decision 的恢复与已执行工具的参数错误也要按实际阶段分别计数。

H7 必须同时区分四种对象：合成业务 output、handler/执行器返回 envelope、模型实际收到的完整 tool body、MemCore 保存/重新投影的 body。允许真实宿主 envelope 存在，但不能把错误层级的 JSON 比较当作保真通过。来源判断必须跟随真实 `call_id`、`source_id` 与 provider 消息边界。

## 3. 分支恢复所需状态

Engine 除 MemCore 外还写入宿主 MemoryStore：用户/助手消息、元数据、session、人格及相关事件等。`final_output_engine` 会读取并可能应用人格状态；`_run_turn_core` 会继续使用原始来源 ID 和时间。只向新 MemCore 重放文本，不恢复宿主状态，不能证明相同实验起点。

| 状态 | 恢复要求 |
| --- | --- |
| 不变配置 | 同一 client mode/capabilities、character pack、人格资源、care=false、domain、时区、模型参数、工具 schema/排序、显式读取权限、compaction 配置。full/card 唯一指定处理差异必须可列举 |
| MemCore 持久状态 | setup 的全部刺激、assistant intermediate/final、工具 action/observation、correlation/source IDs、原始 provider 投影、retention anchor、terminal status、实际 settlement 状态一致 |
| Akane 持久状态 | setup 消息及其源 ID/顺序/时间/元数据、session 与当前 persona 状态等真正影响请求的内容一致；不能只恢复 MemCore |
| 索引 | 从分支自己的保存记录重建/热加载；不得共享可写索引或沿用另一分支新增项。可以复用只读 embedding 模型实例 |
| 回合临时状态 | 新回合重新初始化 open-turn guard、tool history、call-ID 集合、重试/工具计数、request projection state；不能继承上一 probe 的 mutable 列表或计数 |
| 工具与运行时状态 | fixture 可用版本按该 probe 的脚本状态恢复，不能留在 setup v1；能力注册、加载激活、提示上下文状态需一致。无待完成背景任务、未决回合或异步写入再建立 checkpoint |
| 计费和审计状态 | 恢复不发新模型请求，独立记录 replay/clone 操作；费用账本是跨分支共享的权威预算，不随 checkpoint 回滚或重复收费 |

优先做法是关闭或明确排空隔离 setup 宿主，将该实验的新存储状态生成一致快照，再在分离存储中用真实构造重新打开。三个分支可以使用相同逻辑 user/session/character 身份，通过不同物理存储与索引隔离；这样无需篡改 source ID 或原生 namespace 构造。报告须如实说明这种隔离方式，不能称为三个不同的逻辑用户。

当前 `MemcoreManager._get_system` 原生使用空 tenant，user/session/character 映射其它 namespace 字段。仅为让 hash 相同而替换 `_get_system` 或私改 namespace 不是必需手段。若选“原始 SDK 响应经真实 Engine 重放”，必须完整重放所有 setup 的工具 decision、最终恢复尝试和提交，同时保证零网络、同样的业务结果，且最终用 H9 的实际状态比较证明一致。

为确定性重放而限定注入时间/UUID 依赖可以接受，但需满足：正式 setup 与 replay 均使用相同规则；每步单独确定，不能让后台任务消耗共享序列而漂移；不冻结 monotonic timeout、真实请求计费时间或预算时钟；新 probe 的 IDs 不与 setup 冲突。否则即使正文相同，来源或时间依赖问题仍可能不同。单纯生成同形状的假 projection 不算恢复。

## 4. 正式运行前的判定与解释边界

先记录验收产物，再检查正式运行确实使用同一份 `akane_host.py`、同一冻结 Akane/依赖源码和相同通过的注入端口。验收文件需保存所覆盖源码的 hash；正式 runner 在收费前与当前冻结源码逐项比对，不能只接受 `status="passed"` 和 `actual_engine_exercised=true` 两个字段。必须检查真实调用链，而不是把包含 `process_turn(...)` 字符串或写着 `host_equivalent=true` 当证据。尤其排除从 Akane 单元测试复制的裁剪 harness：这些测试可以验证局部函数，却不能单独证明本实验宿主经过完整真实 Engine。

回合通过须有与本次用户刺激同 turn 的新 `message.assistant`、`turn_role="final"` 及实际关闭/提交证据。真实工具 preface 也使用 `message.assistant`，但其角色是 intermediate；单凭“出现了一条新 assistant”不能证明最终回复已提交。`speech` 非空、格式被宿主接受、最终提交成功与语义正确应分别记录。

正式 setup 暴露审阅应覆盖全部 assistant authored 内容、最终 metadata、工具参数，以及协议确实保留的 reasoning 字段；各 probe 的 gold 正确性与来源证据分开。若模型主动提前说出后续答案，保留运行并标记暴露，不抹去回答，也不在无授权改题的前提下删掉那一轮。H10 的 sentinel 检查证明没有系统性注入未来答案，不能替代对正式 setup 自发暴露的语义审阅。

本清单未执行任何真实模型调用，未读取用户运行数据库、日志、环境密钥或凭据。这里只定义并核对源码依据；“宿主等价验收已通过”必须由新宿主的实际离线案例结果另行证明。

## 5. 若采用 Engine 重放：限制非确定字段映射

重放 matcher 应保留原始请求与规范化比较值，输出每一处允许映射的字段路径、来源记录和理由。比较值不能覆盖原始请求，也不能送给正式模型代替真实宿主请求。没有 provenance 的差异直接失败，不能通过全局 UUID、日期或 `source_id` 正则擦除来通过验收。

实际实验的 SDK response replay 保留原始响应 bytes：每次传输的 `replay_id_map` 必须为空，回放响应 SHA256 必须等于源响应 SHA256。来源映射只用于公开快照的结构化 ID 比较，不能让通用 transport 在 matcher 之前替换 request、response 或 SSE event 正文。若原始响应直接引用了无法复用的助手随机 ID，应保留为不支持的 replay 差异，不能静默重写模型的工具参数。

已核对的 ID 生成规则：

| 字段 | 当前真实生成方法 | 比较要求 |
| --- | --- | --- |
| 用户 `source_id` | Engine `_pop_user_memory_source_id`：`plugin-event:` + SHA256；输入含 public `memory_idempotency_key`、profile user、session、character pack、role | 同身份与幂等键重放必须字节相同 |
| 回合 `turn_id` | manager `_stable_turn_id`：`turn:` + 用户 source ID 的 SHA256 前32位 | 用户 ID 稳定时必须字节相同 |
| 原生 provider `call_id` | 来自原始 SDK 响应；缺失时宿主另有签名 fallback | 重放相同 SDK 响应应完全相同，不能任意重命名 |
| 工具 action/result `source_id` | Engine 对 `current_user_source_id\|session_id\|call_id\|tool_type` 做 SHA256 前32位，加 `tooltrace:` 前缀；manager 再加 `:tool_use` / `:tool_result` | 上述输入稳定时两个来源都必须字节相同 |
| 普通助手 final/preface `source_id` | MemoryStore 未传入 source ID 时使用 `str(uuid.uuid4())`；特殊静默回复另有 `assistant_silent_` 前缀 | 只对已经公开记录证实的宿主生成助手来源建立一一映射；不能把工具、用户或正文里出现的 UUID 归入此类 |
| 工具时间 | action 为 `max(payload now_ts, int(time.time()))`，observation 为 action+1 | 只对已核对来源记录的数值进行定向时间映射，验证 action/result 关联及+1关系 |
| 助手 final 时间 | Engine 构造 assistant_record 时使用 `int(time.time())` | 由对应助手来源记录映射；不改模型自己写出的日期或相对时间 |

公共快照可在 SDK 边界只读取得：`MemcoreManager.build_context_projection(...)` 提供 `messages`，每项有 `turn_id`、`payload`、`source_ids`、`payload_hash`、`projection_status`、`projection_index`、`projection_version`。对这些实际返回的 IDs，可调用 manager 的 `open_memory(arguments={"memory_ids": ids, "view": "content", "detail": "full"}, ...)` 批量读取；每个 raw 结果给出 `source_id`、`seq_no`、`timestamp`、`date_label`、`time_of_day`、`kind`、`origin`、`turn_role`、`turn_id`、`correlation_id`、`renderer_id`、`renderer_version`、`content`。不需要读取私有表。原始运行和重放应在同一边界取得摘要，不能只用回合结束快照推测当时尚未产生的 final。

助手来源映射应按稳定 turn、typed role、同轮位置和完全一致的原始内容配对；比较 metadata/关联关系后再登记双向唯一映射。不能只按“第几个 assistant 消息”配对，因为一个 native assistant 消息可能同时覆盖前置 speech 和多个 action source。任何数量差异、重复映射、跨 turn 配对、未知新增 ID 都失败。

只有以下已证实的宿主渲染位置允许在比较副本内定向替换：

1. settled 的 `role="tool"` 卡片，第1行为 `[compact_reloadable]`，第2行为 `time: YYYY-MM-DD HH:MM`。根据该消息的 `tool_call_id` 和唯一 observation source，验证时间行等于该记录 timestamp 在配置时区的实际渲染，再替换该整行。`tool`、`call_id`、`status`、`source_id`、字符/字节长度、`result_hash`、`reload` 等剩余行保持逐字一致。仅发现文本长得像卡片不够，需 projection 元数据和来源记录证明它是宿主投影。
2. 经真实 renderer 生成的 canonical/compact entry 首行可能为 `[YYYY-MM-DD 周X HH:MM | 时段] kind`。仅在 record 的 renderer/kind/turn_role 与整段结构匹配、首行恰好等于各自 timestamp/time_of_day 派生值时映射该首行；所有后续业务正文仍逐字比较。用户来源时间不属于可变映射。
3. 公开投影/读取结果中已登记来源的结构化 ID/time 字段可在比较副本中映射；业务 JSON 里恰好同名的字段不能自动继承此规则。衍生 hash 必须先分别验证其与各自原始 payload 一致，再比较规范化 payload 的新 hash；不能直接忽略原 hash。

普通 native full tool message 的 content 是保存的 output，助手 final 的 content 是 `provider_output_raw`；不能扫描它们并抹去日期、UUID、hash 或数字。记录中的时间不同不表示正文应改变。card 的结果 hash 绑定完整原始正文，也不能作为“易变 hash”清除。

最终基线比较还须覆盖 `host_state` 和投影代数/覆盖指标，不能提取了 session/persona 却在 matcher 中跳过它们。每个 raw source 的 origin、renderer 和可取得的内容/metadata 证据必须进入比较；缺失所需来源、批量 open 中某项失败、同一来源出现互相冲突记录均使比较失败。仅对已定位生成规则的结构化宿主时钟字段放宽，并列出该字段；不通配忽略整个 session、全部 metadata 或所有时间字段。

当前真实 dispatcher 的公开 raw read 不返回 `memory_metadata`，应明确记录该 annotation hash 不可用，不能写成已核验或把空值当作原值。可以用真实公共 `MemoryStore.get_session_messages(...)` 比较宿主消息 metadata，结合逐字保留的模型最终 JSON 与未改动的提交链说明恢复依据；这属于宿主 metadata 证据，不能称为私有 MemCore annotation 逐项比较。卡片的 `projection_content_hash` 也只是可见投影 hash，完整业务正文保真另由 H7 回读证明。

当前 `stable_prefix_hash` 是完整 projection payload 列表的 `stable_projection_hash`；对卡片时间作允许的映射时，先分别验证原始列表与其原 hash 一致，再对规范化列表重算比较。Akane `MemoryStore.add_message` 会用该消息 timestamp 调用 `ensure_session`，后者更新 `session.updated_at`；若该字段实际不同，必须证明它分别对应最后一次真实宿主消息写入，才能随同该来源时间映射。`created_at`、标题、gift focus、人格正文等没有获得这项通配豁免。

在当前已核对的文本回合主线中，没有确认一种应普遍放宽的 `pending_id`。`pending_actions` / `pending_correlations` 是真实生命周期信息，不能当随机 ID 清空。若实际出站请求出现新的 pending 标识，先定位其生成函数与具体字段，再增加最小白名单；不得先写通配规则。所有规范化规则必须有反例测试：只改一个业务时间、旧版本数字、source 关联、工具参数或正文 UUID，即使外形类似宿主字段，matcher 仍拒绝。
