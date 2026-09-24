# 宿主接入代码索引：Akane + MemCore（V1）

日期：2026-09-24。

本文是 [设计理由与研究设计](akane_memcore_design_rationale_and_research_v1.md) 的配套材料，只记录核对过的**代码证据入口**与**版本口径**：哪些文件承担哪些职责、本次读取基于哪个提交。机制论证与研究设计在正文，本文不重复。

本文是 AkaneCompanionLab 这一具体宿主的接入索引，不是 MemCore 的公共 API 契约。其他宿主接入请看 [配置与运行时职责](configuration_api_v1.md) 与 [AI 集成清单](ai_integration_checklist_v1.md)。

## 版本口径

本次读取基于本地工作树：

- MemCore 基准提交：`37be80ebb3e57f814c54d26a68a665e0495b2a85`，相关代码存在既有未提交修改；不能仅凭该提交复现本稿全部观察。
- AkaneCompanionLab 基准提交：`f2579ac0f5b4e63b24307631d054fc4ea50dab40`，工作树也存在既有修改。
- 本稿只读核对代码、文档和测试断言；未启动模型、工具服务或应用，未执行上述实验，未读取私有部署配置和会话数据库。正式实验前应另行固定可复现版本。

## MemCore

| 内容 | 入口 |
| --- | --- |
| turn 提交、结算、provider 历史、目录与来源展开 | [memory_system.py](../memcore/memory_system.py) |
| 卡片生成和收益判断 | [settlement.py](../memcore/settlement.py) |
| 冻结投影、审计与前缀检查 | [projection.py](../memcore/projection.py) |
| token 规划、关系边界、操作摘要、长期强化 | [compaction.py](../memcore/compaction.py) |
| 可见性排除与检索准入 | [retrieval.py](../memcore/retrieval.py) |
| 精确时间页与过大单元边界 | [timeline_read.py](../memcore/timeline_read.py) |
| 目录字段与确定性回退 | [memory_catalog.py](../memcore/memory_catalog.py) |
| metadata 标注及时间提示 | [schema.py](../memcore/schema.py)、[prompts.py](../memcore/prompts.py) |
| 层级存储标记与来源留存 | [sqlite_store.py](../memcore/store/sqlite_store.py) |

## AkaneCompanionLab

下列文件名相对于 AkaneCompanionLab 仓库根目录，不要求两个项目位于固定本机路径。行号是本次核对的定位提示，后续版本应以函数和职责重新检索。

| 内容 | 入口与定位提示 |
| --- | --- |
| 工具执行、实际反馈写入、开放轮历史 | `companion_v01/engine.py`，约 8000、8282、8589 行 |
| final 原文、标注与语义正文提交 | `companion_v01/memcore_integration/manager.py`，约 1758 行 |
| 宿主配置传入 MemCore | 同上，`_build_memory_config`；`config.py` |
| 结果页透传，不进行统一字符截断 | `companion_v01/tool_orchestration_engine.py`，约 222 行 |
| 角色状态、动态内容与历史组织 | `companion_v01/prompt_builder.py`，约 281—350 行 |
| speech、标注及产品状态的处理 | `companion_v01/final_output_engine.py`，约 142—178 行 |
| Skill 正文按需读取 | `companion_v01/skill_runtime.py`，约 395 行；`companion_v01/tool_handlers/skills.py` |
| 能力目录本代 base 与最新 update | `companion_v01/memcore_integration/manager.py`，约 1525 行；`companion_v01/engine_services/response_builder.py`，约 67 行 |
| MCP schema 扩展与历史别名路由 | `companion_v01/engine_services/tool_rounds.py`，约 323、794 行 |
| 已入时间线贡献的重复注入控制 | `companion_v01/prompt_context_lifecycle.py`，约 44 行 |
