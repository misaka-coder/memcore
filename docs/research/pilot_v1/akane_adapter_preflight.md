# Akane 实验接入检查

核对日期：2026-09-06。下述 PromptBuilder 最小接入已实际完成，生产 embedding 与四组真实场景也已运行，见 [首轮报告](first_real_experiment_report.md)。本文件保留完整宿主入口的源码说明；完整 Akane Engine 尚未运行。

Akane 根目录的 config.py 会在导入时解析/创建数据目录，并读取 AKANE_ENV_FILE；该变量为空仍回退 .env。因此实验必须在独立子进程启动前同时设置新的 AKANE_DATA_ROOT 和指向实验空配置文件的 AKANE_ENV_FILE，再导入宿主模块。不能只换 MemCore namespace。实验材料不复制日常数据库、日志、缓存或凭据。

新实验根中可准备 instances/research-preflight/instance.toml：

~~~toml
schema_version = 1
instance_id = "research-preflight"
character_pack_id = "akane_v1"
plugins = []

[features]
care = false

[channels.qq]
enabled = false
~~~

当前完整启动经 host_bot_bootstrap 到 BotRuntimeFactory.create；工厂先绑定独立数据根，再加载根内配置。新根不配置 bots.toml 时可走单实例兼容路径。若启动 HTTP 宿主，使用独立实验管理 token。features.care 是严格布尔字段；disabled CareModule 路径会过滤养成状态并跳过对应更新，仍需核对实际请求。

本轮实际复用 PromptBuilder(load_persona_config()) 与 PromptProfileRegistry().get(ClientMode.DESKTOP_PET, care_enabled=False)，把 profile 的 system_prompt_override、mode_prompt_override 交给 build_final_generation_context。16个步骤各构建两次，稳定系统和人格一致，care规则实际缺席；固定人格、视觉默认值和脚本时间，两组的动态资源输入均为空。

Builder 分开提供 system_prompt、system_extra_blocks、history_turns、user_prompt、ephemeral_turns；不要压成单个字符串后重拼。本次独立提交的用户消息应从 history_turns 排除，临时状态不能写成长期历史。MemCore build_context_projection 已包含当前刺激，拼接时继续检查当前消息只出现一次。

工具可先使用 MemCore 公共原生规格与 dispatcher，另加受控 lookup_fixture。若验证 Akane 的转换，入口是 native_tool_schema.build_openai_native_tool_specs 的 allowed_tool_names；完整轮次入口是 AkaneMemoryEngine.process_turn。LLMRuntime.call_chat_json_result 保留结果信息，不能只取 parsed 或把 fallback 当成功响应。每次实际 provider 请求仍需接同一收费账本，保存模型实际收到的完整结果。

复用 PromptBuilder 只能称为“复用宿主提示契约”。实际运行 Engine 的配置、提示、工具、提交与隔离链路后，才能称为 Akane 端到端验收。旧 E5 多实例 smoke 使用 legacy memory 与 hashed embedding，验证的是隔离，不是本研究的生产配置。

已实际加载本地 BAAI/bge-m3（1024维、CUDA），正式进程重复健康检查通过，无哈希回退。四组模型自主工具循环、十二个同条件历史分支恢复、答案暴露审计与逐例评分均已完成；包含七个失败探针，详见首轮报告。
