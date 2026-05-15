# 焓变计算方案（修正版）

## 物理定义

**参考态 (Reference)**: "as-cooled"态 —— 样品从 200°C 以设定冷速直接冷却至室温（**不经任何等温退火**），再以 10°C/min 升温测得的 DSC 曲线。

**退火态 (Annealed)**: 样品在退火温度下等温保持不同时间后，冷却至室温，再升温测得的 DSC 曲线。

> **注意**：参考态是一个**独立的空白实验**，不是现有实验中的任何一个 ramp。需要单独测量后提供。

## 计算流程

### Step 1: 构建 ΔDSC

```
DSC_empty(T) = 空坩埚基线 (µW)
DSC_ref(T)   = 参考态升温段原始 DSC (µW)  [as-cooled, 待测量]
DSC_i(T)     = 第 i 个退火态升温段原始 DSC (µW)

ΔDSC_ref(T) = DSC_ref(T) − DSC_empty(T)
ΔDSC_i(T)   = DSC_i(T) − DSC_empty(T)
```

### Step 2: 构建 ΔCp 曲线

```
ΔCp_i(T) ∝ ΔDSC_i(T) − ΔDSC_ref(T)
          = (DSC_i − DSC_empty) − (DSC_ref − DSC_empty)
          = DSC_i(T) − DSC_ref(T)
```

物理含义：退火态与参考态的热流差值，即纯粹的弛豫吸热信号。

### Step 3: 积分

```
T_low  = 50 °C    (室温以上，避免起始不稳)
T_high = T_onset  (过冷液态起点，ΔCp 归零处)

Δh_i = (1/(β·m)) × ∫[T_low, T_high] (ΔDSC_i − ΔDSC_ref) dT

     = (6/4700) × ∫[50, T_onset] (DSC_i − DSC_ref) dT   [J/g]
```

**单位换算**：
- β = 10°C/min = 1/6 °C/s
- m = 4.7 mg = 0.0047 g
- 1 / (β × m × 1000) = 6 / 4700 = 0.0012766 (µW·°C → J/g)

### Step 4: T_onset 确定方法

对每个 ramp 的 DSC_i(T) 曲线：
1. 在 110-160°C 区间线性拟合液态基线: DSC_liq(T) = a·T + b
2. 在 70-115°C 区间找到 overshoot 峰值
3. 从峰值向后扫描，找到 DSC 信号首次回到液态基线的 crossover 温度
4. 该温度即为 T_onset（过冷液态起点，ΔCp 在此归零）

## 待提供的数据

| 实验 | 参考态 | 说明 |
|------|--------|------|
| Onestep 50°C | 待测量 | 200→50°C 冷却(无保温)→30°C→升温 |
| Onestep 70°C | 待测量 | 200→70°C 冷却(无保温)→30°C→升温 |
| Twosteps | 待测量 | 200→90→80°C 冷却(无保温)→30°C→升温 |
| Kovacs | 待测量 | 200→80→90°C 冷却(无保温)→30°C→升温 |

## 验证

- Δh_i 应为正值（弛豫释放焓 > 0）
- Δh_i 随退火时间单调增大
- 参考态自洽: 若将参考态作为退火态输入，Δh_ref ≡ 0
