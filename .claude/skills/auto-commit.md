---
name: auto-commit
description: Auto-organize code and results, then commit and push to GitHub Richardlei branch
model: haiku
---

# Auto Commit & Push

每次编程完成后自动执行：整理代码和结果文件，提交并推送到 GitHub 的 Richardlei 分支。

## 执行流程

1. 检查项目目录 `/root/twoannealing` 是否有未提交的更改
2. 自动 stage 所有更改（代码、结果、数据文件）
3. 生成带时间戳的提交信息
4. 推送到 `origin Richardlei`

## 触发方式

此技能通过 **Stop hook** 自动触发，无需手动调用。每次 Claude Code 会话结束时自动执行。

## 手动调用

如需手动触发，使用 `/auto-commit` 命令。
