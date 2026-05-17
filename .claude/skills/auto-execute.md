---
name: auto-execute
description: Auto-execute commands without permission prompts; only ask user for approach decisions
model: haiku
---

# Auto Execute Mode

所有命令自动执行，不再询问权限。仅在方案选择/设计决策时咨询用户。

## 行为说明

- **Read / Edit / Write** — 自动执行，不弹窗
- **Bash 命令** — sandbox 内自动执行（已在 sandbox 配置中限制文件访问范围）
- **WebSearch / WebFetch** — 自动执行
- **方案选择** — 当有多种实现方式时，仍然会询问用户偏好

## 依赖配置

此技能依赖 `permissions.defaultMode: "dontAsk"` 和 sandbox 限制，已在 settings.local.json 中配置。
