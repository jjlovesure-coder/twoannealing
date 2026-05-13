# DSC 数据文件内容摘要

## 目录结构

```
data/
├── csv/          # 转换后的 CSV 数据文件（可直接导入分析软件）
├── raw/          # 原始仪器输出文件（xlsx / txt）
└── README.md     # 本文件
```

---

## 文件清单

### 1. baseline-01 — DSC 基线测量

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/baseline-01.xlsx` |
| CSV 文件 | `csv/baseline-01.csv` (8,481 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS-empty-02 |
| 样品 | BASELINE (空坩埚基线) |
| 测量时间 | 2026-05-09 18:16 |
| 操作者 | lxf |
| 温度程序 | 30 °C → 200 °C, 升温速率 10 °C/min |
| 气氛 | N₂ 30 ml/min |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 2. reference-01 — DSC 参比测量（蓝宝石）

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/reference-01.xlsx` |
| CSV 文件 | `csv/reference-01.csv` (10,440 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS-ref-01 |
| 样品 | sapphire (蓝宝石标准参比, 10 mg) |
| 测量时间 | 2026-05-09 13:01 |
| 操作者 | lxf |
| 温度程序 | 30 °C → 200 °C, 升温速率 10 °C/min |
| 气氛 | N₂ 30 ml/min |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 3. onestep-01 — 一步退火 (退火温度 50 °C)

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/onestep-01.xlsx` |
| CSV 文件 | `csv/onestep-01.csv` (250,642 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS-onestep-01 |
| 样品 | PS (聚苯乙烯, 4.7 mg) |
| 测量时间 | 2026-05-06 07:48 |
| 温度程序 | 3 次循环，退火温度 50 °C |
| | Step 1: 200→50 °C @ 60 °C/min, 等温 1 min |
| | Step 2: 50→30 °C @ 60 °C/min, 等温 1 min |
| | Step 3: 30→200 °C @ 10 °C/min, 等温 0.1 min |
| | 循环 1 降温速率: 0.1 °C/min (Step 2 前等温) |
| | 循环 2 降温速率: 0.5 °C/min |
| | 循环 3 降温速率: 0.8 °C/min |
| 气氛 | N₂ 30 ml/min |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 4. onestep-02 — 一步退火 (退火温度 70 °C)

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/onestep-02.xlsx` |
| CSV 文件 | `csv/onestep-02.csv` (250,649 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS-onestep-02 |
| 样品 | PS (聚苯乙烯, 4.7 mg) |
| 测量时间 | 2026-05-06 21:03 |
| 温度程序 | 3 次循环，退火温度 70 °C |
| | Step 1: 200→70 °C @ 60 °C/min, 等温 1 min |
| | Step 2: 70→30 °C @ 60 °C/min, 等温 1 min |
| | Step 3: 30→200 °C @ 10 °C/min, 等温 0.1 min |
| | 循环 1 降温速率: 0.1 °C/min |
| | 循环 2 降温速率: 0.5 °C/min |
| | 循环 3 降温速率: 0.8 °C/min |
| 气氛 | N₂ 30 ml/min |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 5. twostep-01 — 两步退火 (退火温度 90 °C → 80 °C)

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/twostep-01.xlsx` |
| CSV 文件 | `csv/twostep-01.csv` (255,778 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS-02 |
| 样品 | PS (聚苯乙烯, 4.7 mg) |
| 测量时间 | 2026-04-30 08:25 |
| 温度程序 | 3 次循环，退火温度 90 °C → 80 °C |
| | Step 1: 200→90 °C @ 60 °C/min, 等温 1 min |
| | Step 2: 90→80 °C @ 60 °C/min, 等温 1 min |
| | Step 3: 80→30 °C @ 60 °C/min, 等温 1 min |
| | Step 4: 30→200 °C @ 10 °C/min, 等温 0.1 min |
| | 循环 1 Step 2 降温速率: 0.1 °C/min |
| | 循环 2 Step 2 降温速率: 0.5 °C/min |
| | 循环 3 Step 2 降温速率: 0.8 °C/min |
| 气氛 | N₂ 30 ml/min |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 6. kovacs-01 — Kovacs 效应实验

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/kovacs-01.xlsx` |
| CSV 文件 | `csv/kovacs-01.csv` (256,421 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS-kovacs-01 |
| 样品 | PS (聚苯乙烯, 4.7 mg) |
| 测量时间 | 2026-05-03 21:56 |
| 温度程序 | 4 次循环，退火温度 90 °C → 80 °C (Kovacs 协议) |
| | Step 1: 200→80 °C @ 60 °C/min, 等温 1 min |
| | Step 2: 80→90 °C @ 60 °C/min, 等温 1 min |
| | Step 3: 90→30 °C @ 60 °C/min, 等温 1 min |
| | Step 4: 30→200 °C @ 10 °C/min, 等温 0.1 min |
| | 循环 1 Step 2 升温速率: 0.1 °C/min |
| | 循环 2 Step 2 升温速率: 0.5 °C/min |
| | 循环 3 Step 2 升温速率: 0.8 °C/min |
| | 循环 4 Step 2 升温速率: 0.1 °C/min |
| 气氛 | N₂ 30 ml/min |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 7. ps-60-hitachi — PS60 热历史重复实验 (Hitachi 仪器)

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/ps-60-hitachi.xlsx` |
| CSV 文件 | `csv/ps-60-hitachi.csv` (17,498 行, 4 列) |
| 仪器模块 | DSC (Hitachi) |
| 数据名 | PS60-01 |
| 样品 | PS (聚苯乙烯, 4.7 mg) |
| 测量时间 | 2026-04-29 16:52 |
| 操作者 | lxf |
| 温度程序 | 200↔30 °C 循环 5 次, 升降温速率均为 60 °C/min |
| 气氛 | N₂ 30 ml/min |
| 实验类型 | 热历史重复实验 |
| CSV 列 | Time/min, Temp/°C, DSC/μW, DDSC/(μW/min) |

---

### 8. ps-60-Mettler — PS60 热历史重复实验 (Mettler Toledo 仪器)

| 属性 | 值 |
|---|---|
| 原始文件 | `raw/ps-60-Mettler.txt` (UTF-16 编码) |
| CSV 文件 | `csv/ps-60-Mettler.csv` (1,751 行, 5 列) |
| 仪器 | Mettler Toledo DSC (STARe SW 14.00) |
| 样品 ID | 2026004715 |
| 测量时间 | 2026-05-06 16:11 |
| 实验类型 | 热历史重复实验 |
| CSV 列 | Index, t/s, Ts/°C, Tr/°C, Value/(W·g⁻¹) |

---

## 实验概览

| 实验类型 | 文件 | 仪器 | 样品 | 数据量 |
|---|---|---|---|---|
| 基线 | baseline-01 | Hitachi | 空坩埚 | 8,481 |
| 参比 | reference-01 | Hitachi | 蓝宝石 10mg | 10,440 |
| 一步退火 50°C | onestep-01 | Hitachi | PS 4.7mg | 250,642 |
| 一步退火 70°C | onestep-02 | Hitachi | PS 4.7mg | 250,649 |
| 两步退火 90→80°C | twostep-01 | Hitachi | PS 4.7mg | 255,778 |
| Kovacs 效应 | kovacs-01 | Hitachi | PS 4.7mg | 256,421 |
| 热历史重复 | ps-60-hitachi | Hitachi | PS 4.7mg | 17,498 |
| 热历史重复 | ps-60-Mettler | Mettler Toledo | PS (ID:2026004715) | 1,751 |

**注**: 所有 Hitachi DSC 数据包含 4 列 (Time, Temp, DSC, DDSC)；Mettler 数据包含 5 列 (Index, t, Ts, Tr, Value)。
