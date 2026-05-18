# 焓变计算与TNM模型拟合 — 完整流程与原理

## 一、实验设计

### 1.1 Kovacs上跳实验 (80°C→90°C)

| 步骤 | 温度 | 操作 |
|------|------|------|
| 1 | T0 (玻璃态) | 起始假想温度，编码热历史 |
| 2 | T1=80°C | 快速淬冷至T1，恒温等待 **t₁**=50s 或 500s |
| 3 | T2=90°C | 上跳至T2，恒温等待 **t₂**=0.1~1000s (10个时间点) |
| 4 | →Tg以下 | 快速降至玻璃态温度，冻结结构 |
| 5 | 升温扫描 | 10°C/min从10°C升到180°C，DSC记录 |

### 1.2 Two-Step下跳实验 (90°C→80°C)

| 步骤 | 温度 | 操作 |
|------|------|------|
| 1 | T1=90°C | 高温平衡态，恒温等待 **t₁**=50s 或 500s |
| 2 | T2=80°C | 下跳至T2，恒温等待 **t₂**=0.1~1000s (10个时间点) |
| 3 | →Tg以下 | 快速降至玻璃态温度，冻结结构 |
| 4 | 升温扫描 | 10°C/min升到180°C，DSC记录 |

**关键物理现象（Kovacs记忆效应）**：在Kovacs上跳实验中，t₁较长的样品(t₁=500s)在短t₂时反而显示
出**更大**的ΔH。这是因为t₁越长→Tf越低→上跳后τ越小→弛豫越快→焓恢复越快。

---

## 二、Stage 1: 原始DSC数据 → 焓变ΔH

### 2.1 数据输入与预处理

**文件**：`src/export_enthalpy.py`

**输入**：
- 空坩埚基线：`data/ps-empty-01.xlsx`（线性插值函数）
- 实验DSC数据：Excel文件，包含Time、Temp、DSC(µW)、DDSC四列
- 样品质量：`M_SAMPLE = 4.7 mg`

**DSC升温段检测**（`export_enthalpy.py:81`）：
通过滑窗计算dT/dt，检测速率>4 K/min、持续>500采样点、温度跨度≥150°C的升温段
（从40°C以下跨到175°C以上）。

### 2.2 核心公式1：ΔH积分计算

**转换因子**（`export_enthalpy.py:19-23`）：

$$\beta = \frac{dT}{dt} = \frac{10\text{ K/min}}{60\text{ s/min}} = \frac{1}{6}\text{ K/s}$$

$$\text{CONV\_JG} = \frac{1/\beta}{m \times 1000} = \frac{6}{4.7 \times 1000} = 1.277 \times 10^{-3}\text{ J/g per µW·°C}$$

**积分范围**：T_min=40°C 到 T_onset（液体起始温度，~103°C）

**积分公式**：

$$\int_{40°C}^{T_{onset}} \Delta DSC(T) \, dT = \int_{40°C}^{T_{onset}} [DSC_{sample}(T) - DSC_{empty}(T)] \, dT \quad [\mu W \cdot °C]$$

**焓变计算**：

$$\Delta H_i = \left(\int_{40}^{T_{onset,i}} \Delta DSC \, dT - \text{ref}\right) \times \text{CONV\_JG}$$

其中：
- **Kovacs实验**：ref = 外部参考扫描积分值（`ps-ref-02.xlsx`），代表未老化玻璃的基线焓
- **Two-Step实验**：ref = 第一个升温段的积分值（最短t₂的曲线作为内参）
- 符号惯例：ΔH > 0 表示焓恢复（结构弛豫放热）

### 2.3 T_onset检测

**文件**：`export_enthalpy.py:109`

**方法**：
1. 在110-160°C范围的液体区做DSC vs T线性拟合 → 液相基线
2. 在70-115°C范围找DSC峰值位置
3. 从峰后向高温方向搜索：DSC(T) − 液相基线(T) 首次由正变负的交叉点即为T_onset

---

## 三、Stage 2: KWW唯象拟合

### 3.1 KWW公式

**文件**：`src/kovacs_fit_phenom.py`

**核心方程 — 时间平移Kohlrausch-Williams-Watts拉伸指数**（`kovacs_fit_phenom.py:50`）：

$$\Delta H(t; g) = H_{max} - H_{max} \cdot \exp\left[-\left(\frac{t + c_g}{\tau_g}\right)^{\beta_g}\right]$$

其中：
- **H_max** [J/g]：共享渐近值，T=T₂时的平衡焓恢复量（两组共用）
- **τ_g** [s]：特征弛豫时间（g=50s组或500s组）
- **β_g** [−]：拉伸指数 (0<β≤1)，β越小分布越宽
- **c_g** [s]：等效时间偏移，编码t=0时的初始条件

**c_g的计算**（`kovacs_fit_phenom.py:82`）：

$$c_g = \tau_g \cdot \left[-\ln\left(1 - \frac{\Delta H_{0,g}}{H_{max}}\right)\right]^{1/\beta_g}$$

物理意义：ΔH₀,g是t₂=0时的"表观"初始焓值。由于t₁阶段已经发生弛豫，实际起始点的
KWW等效年龄为c_g秒。

### 3.2 参数向量

| 参数 | 含义 | 典型范围 |
|------|------|----------|
| H_max | 共享平衡渐近值 | 5-10 J/g |
| ΔH₀,50s | 50s组初始焓 | −1 到 3 J/g |
| τ_50s | 50s组特征时间 | 30-3000 s |
| β_50s | 50s组拉伸指数 | 0.1-0.9 |
| ΔH₀,500s | 500s组初始焓 | 1.5-5 J/g |
| τ_500s | 500s组特征时间 | 30-3000 s |
| β_500s | 500s组拉伸指数 | 0.1-0.9 |

### 3.3 优化方法

**两阶段优化**（`kovacs_fit_phenom.py:88`）：
1. **全局搜索**：scipy `differential_evolution`（进化/遗传算法变体），maxiter=5000
2. **局部精修**：scipy `least_squares`（TRF方法，Trust Region Reflective）

**代价函数**：残差平方和 (SSE)

$$J = \sum_{i} (\Delta H_{pred,50s}(t_i) - \Delta H_{exp,50s,i})^2 + \sum_{j} (\Delta H_{pred,500s}(t_j) - \Delta H_{exp,500s,j})^2$$

### 3.4 输出

- 稠密采样曲线：`results/kovacs_phenom_curve.csv` (150个对数间隔点，t从0.1到10^8 s)
- 用于Stage 3的TNM逆向工程

---

## 四、Stage 3: TNM物理模型拟合

### 4.1 TNM模型核心方程

**文件**：`src/kovacs_reverse_tnm.py`

**Tool-Narayanaswamy-Moynihan模型**描述玻璃的结构弛豫，核心变量是假想温度
(fictive temperature) Tf。

#### 弛豫时间 τ(T, Tf)

$$\tau(T, T_f) = A \cdot \exp\left(\frac{x \cdot H^*}{R \cdot T} + \frac{(1-x) \cdot H^*}{R \cdot T_f}\right)$$

| 参数 | 含义 | 单位 |
|------|------|------|
| **A** | 指前因子 | s |
| **H\*** | 表观活化能 | J/mol |
| **x** | 非线性参数 (0<x≤1) | – |
| **T** | 实际温度 | K |
| **Tf** | 假想温度 | K |
| **R** | 气体常数 = 8.314 | J/(mol·K) |

**x的物理意义**：
- x→1：纯Arrhenius行为，τ仅由实际温度决定
- x→0：纯结构控制，τ仅由假想温度决定
- 典型聚合物值：x≈0.3-0.95（PS: x≈0.9，见D'Amore 2006）

#### 等温弛豫 (KWW形式)

TNM的标准等温解法使用拉伸指数：

$$T_f(t) = T + (T_{f,start} - T) \cdot \exp\left[-\left(S(t)\right)^\beta\right]$$

其中**约化时间** S(t) 通过数值积分计算：

$$S(t) = \int_0^t \frac{dt'}{\tau(T, T_f(t'))}$$

这需要**自洽迭代求解**：当前Tf决定τ，τ决定S，S决定新Tf。

#### 数值实现

**文件**：`kovacs_reverse_tnm.py:56`

使用12个对数间隔的时间步长进行迭代积分：

```python
for t_i in t_steps:
    τ = A * exp(x·H*/(RT) + (1-x)·H*/(RTf))
    S += (t_i - t_prev) / τ
    Tf = T + (Tf_start - T) * exp(-(S)^β)
```

**瞬时淬冷假设**：从T0降至T1的冷却过程被完全吸收到参数T0中（即：不考虑有限的降温速率）。
这是聚合物文献的标准做法（参见 D'Amore 2006, Grassia 2018）。

### 4.2 Kovacs协议模拟

**文件**：`kovacs_reverse_tnm.py:75`

```
T0 →[quench]→ T1(80°C), hold t1
   →[quench]→ T2(90°C), hold t2
```

**归一化焓变**：

$$\Delta H_{norm} = \frac{T_0 - T_f(t_2)}{T_0 - T_2}$$

物理含义：(T0 − Tf)是结构偏离平衡的程度，归一化后ΔH_norm ∈ [0, 1]。

含标度因子**scale**的完整表达式：

$$\Delta H(t_2) = \text{scale} \times \frac{T_0 - T_f(t_2)}{\max(T_0 - T_2, 1)}$$

### 4.3 两种TNM拟合路线

#### 路线A：逆向工程 (从KWW→TNM)

**文件**：`src/kovacs_reverse_tnm.py`

- 目标数据：`kovacs_phenom_curve.csv`（KWW稠密采样曲线，150点）
- 优化器：多起点L-BFGS-B (40起点)
- 参数：`[logA, H*, x, β, T0, scale]`
- Kovacs结果：R²_50s=0.946, R²_500s=0.992

#### 路线B：直接拟合 (实验数据→TNM)

**文件**：`src/twosteps_fit_tnm.py` / `src/ga_fit_tnm.py`

- 目标数据：`enthalpy_twosteps.csv` 或 `enthalpy_kovacs.csv`（原始实验点，20点）
- 优化器：多起点L-BFGS-B / 遗传算法 / 微分进化
- 代价函数：SSE（与实验ΔH的残差平方和）

---

## 五、遗传算法/Memetic GA拟合

### 5.1 算法架构

**文件**：`src/ga_fit_tnm.py`

#### Memetic GA组件

| 组件 | 实现 | 参数 |
|------|------|------|
| **编码** | 实数编码（6维向量） | [logA, H*, x, β, T0, scale] |
| **种群** | 3岛模型 | 各60个体，共180 |
| **选择** | 锦标赛选择 | k=3 |
| **交叉** | BLX-α交叉 + DE差分变异 | α=0.3 |
| **变异** | 自适应高斯变异 | 初始σ=0.30×range, 指数衰减至0.03 |
| **精英** | 保留前8% | ~5个/岛 |
| **迁移** | 每20代跨岛迁移最优2个体 | – |
| **Memetic** | 每15代对全局最优做L-BFGS-B | maxiter=80 |
| **重启** | 停滞60代后最优解附近重初始化 | σ=0.05×range |

#### 自适应变异率

$$\mu(gen) = \mu_{init} \cdot \left(\frac{0.03}{\mu_{init}}\right)^{gen/G} + 0.03$$

其中 μ_init = 0.30, G = 500，即从30%指数衰减至3%。

#### BLX-α交叉

对每维i，子代从扩展区间均匀采样：

$$child_i \sim \mathcal{U}\left[\min(p1_i, p2_i) - \alpha \cdot \Delta, \;\max(p1_i, p2_i) + \alpha \cdot \Delta\right]$$

其中 Δ = |p1_i − p2_i|, α=0.3。

#### DE差分变异（混合算子，与BLX-α交替使用）

$$child = p_1 + F \cdot (p_2 - p_3), \quad F \sim \mathcal{U}(0.3, 0.9)$$

DE概率随代数线性增长（0→0.3），在后期替代变弱的BLX-α交叉。

### 5.2 参数边界

| 参数 | Kovacs边界 | Two-Step边界 |
|------|-----------|-------------|
| logA | [−36, −8] | [−35, −15] |
| H* [J/mol] | [80000, 350000] | [100000, 400000] |
| x | [0.1, 0.95] | [0.2, 0.99] |
| β | [0.1, 0.7] | [0.1, 0.99] |
| T0 [K] | [375, 420] | [380, 460] |
| scale [J/g] | [1.0, 15.0] | [3.0, 10.0] |

### 5.3 优化结果对比

#### Kovacs上跳 (80°C→90°C) — 良好约束问题

| 方法 | Cost | RMSE | R²(50s) | R²(500s) | 时间 |
|------|------|------|---------|----------|------|
| **Memetic GA** | 1.7633 | 0.2969 | 0.9524 | 0.9851 | 396s |
| Diff.Evolution | 1.7633 | 0.2969 | 0.9524 | 0.9851 | 1505s |
| Multi-BFGS | 1.7702 | 0.2975 | 0.9519 | 0.9859 | 198s |

**最优参数**：

| 参数 | 值 |
|------|-----|
| logA | −31.48 |
| A | 3.33×10⁻³² s |
| H* | 235.8 kJ/mol |
| x | 0.950 |
| β | 0.700 |
| T0 | 383.6 K (110.5°C) |
| scale | 6.27 J/g |

#### Two-Step下跳 (90°C→80°C) — 弱约束问题

| 方法 | Cost | RMSE | R²(50s) | R²(500s) | 时间 |
|------|------|------|---------|----------|------|
| Diff.Evolution | 0.9530 | 0.2183 | 0.8200 | 0.7885 | 1669s |
| Multi-BFGS | 0.9530 | 0.2183 | 0.8200 | 0.7882 | 156s |
| Memetic GA | 0.9688 | 0.2201 | 0.8173 | 0.7828 | 370s |

所有方法收敛到边界解（x=0.99, β=0.99, T0=460K），说明Two-Step数据不足以唯一约束
6个TNM参数——需要更多样的实验条件或简化模型（如固定x=1简化为Arrhenius极限）。

---

## 六、完整公式汇编

### 6.1 DSC→焓变

$$\Delta H_i = \left[\int_{40°C}^{T_{onset,i}} (DSC_{samp} - DSC_{empty})\,dT - \text{ref}\right] \times \frac{1/\beta}{m \times 1000}$$

$$\beta = \frac{10\text{ K/min}}{60} = 0.1667\text{ K/s}, \quad m = 4.7\text{ mg}$$

### 6.2 KWW唯象模型

$$\Delta H(t) = H_{max} - H_{max} \cdot \exp\left[-\left(\frac{t + c}{\tau}\right)^\beta\right]$$

$$c = \tau \cdot \left[-\ln\left(1 - \frac{\Delta H_0}{H_{max}}\right)\right]^{1/\beta}$$

### 6.3 TNM物理模型

**弛豫时间**:

$$\tau(T, T_f) = A \cdot \exp\left(\frac{x H^*}{RT} + \frac{(1-x) H^*}{RT_f}\right)$$

**等温弛豫 (瞬时淬冷)**:

$$T_f(t) = T + (T_{f,0} - T) \cdot \exp\left[-\left(\int_0^t \frac{dt'}{\tau(T, T_f(t'))}\right)^\beta\right]$$

**Kovacs上跳模拟**:

$$T_f(t_2) = f\Big(T_2, t_2,\; f(T_1, t_1, T_0)\Big)$$

$$\Delta H = \text{scale} \times \frac{T_0 - T_f}{\max(T_0 - T_2, 1)}$$

### 6.4 优化代价函数

$$J = \sum_{i=1}^{N_{50}} (\Delta H_{pred,50s} - \Delta H_{exp,50s})^2 + \sum_{i=1}^{N_{500}} (\Delta H_{pred,500s} - \Delta H_{exp,500s})^2$$

$$RMSE = \sqrt{\frac{J}{N_{total}}}, \quad R^2 = 1 - \frac{\sum(y_{pred} - y_{exp})^2}{\sum(y_{exp} - \bar{y}_{exp})^2}$$

---

## 七、关键文件索引

| 文件 | 功能 |
|------|------|
| `src/export_enthalpy.py` | DSC原始数据→ΔH提取 (Stage 1) |
| `src/kovacs_fit_phenom.py` | Kovacs KWW唯象拟合 (Stage 2) |
| `src/twosteps_fit_phenom.py` | Two-Step KWW唯象拟合 |
| `src/kovacs_reverse_tnm.py` | 从KWW逆向工程TNM参数 (Stage 3, 路线A) |
| `src/twosteps_fit_tnm.py` | TNM直接拟合实验数据 多起点BFGS (Stage 3, 路线B) |
| `src/ga_fit_tnm.py` | TNM遗传算法/Memetic GA拟合 (Stage 3, 路线B) |
| `src/run_full_workflow.py` | Kovacs完整工作流编排 (Stage 1→2→3→4) |
| `src/run_twosteps_workflow.py` | Two-Step完整工作流编排 |
| `src/plot_raw_curves.py` | DSC原始曲线绘图 |

### 输出文件

| 文件 | 内容 |
|------|------|
| `results/enthalpy/enthalpy_kovacs.csv` | Kovacs实验提取的ΔH |
| `results/enthalpy/enthalpy_twosteps.csv` | Two-Step实验提取的ΔH |
| `results/kovacs_phenom_curve.csv` | KWW稠密采样曲线 |
| `results/tnm/tnm_reverse_params.csv` | TNM逆向工程参数 |
| `results/tnm/tnm_twosteps_params.csv` | TNM直接拟合参数 |
| `results/tnm/ga_all_params.csv` | 遗传算法全部参数比较 |
| `results/kovacs_full_comparison.png` | Kovacs最终三向对比图 |
| `results/twosteps_full_comparison.png` | Two-Step最终三向对比图 |

---

## 八、关键物理结论

1. **Kovacs上跳数据约束力强**：TNM模型能很好地重现记忆效应（R²>0.95），参数值符合
   PS的文献值（x≈0.95, β≈0.7, logA≈−31, H*≈230 kJ/mol）

2. **Two-Step下跳数据约束力弱**：6个参数中有3个（x, β, T0）碰到边界，说明需要更多样
   的实验条件（如不同T1/T2温度组合、不同t₁时间）或考虑简化模型（如固定x→1的
   Arrhenius极限）

3. **Memetic GA与Differential Evolution性能相当**：在良好约束问题（Kovacs）上两者都
   找到全局最优（cost=1.7633），且GA快3.8倍（396s vs 1505s）

4. **Multi-start BFGS最快但不可靠**：198s完成但陷入局部极小值（cost=1.770 vs 全局最优1.763），
   在强非线性参数空间中缺乏全局搜索能力

5. **瞬时淬冷假设的局限**：当前TNM模型假定从T0→T1→T2的冷却为无限快速，有限降温速率
   （60 K/min）的效应被吸收到T0参数中。更完整的模型应考虑非等温淬冷路径
