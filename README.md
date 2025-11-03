# SymbolicPlastic-SNN

符号突触 + 结构可塑性 的大规模脉冲网络（无梯度训练，支持固定点模式）

## 规格与约定

- 项目名：`symbolicplastic_snn`
- 默认技术栈：Python 3.11 + NumPy（纯 CPU，便于快速集成与测试；后续可替换为 C++/Rust 内核）
- 依赖约束：仅允许 `numpy`、`pytest`；禁止其他外部依赖
- 代码风格：PEP8 + 类型标注 + `dataclasses`，纯函数优先
- 统一 PRNG：`xorshift64*`（内置实现），相同 seed + 相同输入 → 完全确定性
- 固定点默认格式：
  - 膜电位 `v`、阈值 `θ`：`int16`（Q4.11）
  - 泄露 `λ`：`uint16`（Q1.15）
  - 事件累加：`int32`
  - 概率：`uint16`（Q0.16）

模块布局（逐步填充）：

```
symbolicplastic_snn/
  __init__.py
  core/lif_fixedpoint.py
  schedule/timewheel.py
  conn/alias.py
  conn/permute.py
  realtime/budgeter.py
  plasticity/stats.py
  plasticity/update.py
  readout/readout.py
  io/config.py
  io/snapshot.py
  encode/input_encoders.py
  runner/loop.py
tests/
  ...（与模块一一对应）
```

统一 PRNG（xorshift64*）说明：

- 内置实现，所有随机过程（拓扑、别名采样、抖动等）统一调用；确保相同 seed + 相同输入下完全确定性。
- 参考实现（Python 伪代码）：

```python
from typing import Iterator

def xorshift64star(seed: int) -> Iterator[int]:
    """64-bit xorshift* PRNG, returns 64-bit unsigned integers.
    Multiplier 0x2545F4914F6CDD1D is the Marsaglia suggested constant.
    """
    x = seed & 0xFFFFFFFFFFFFFFFF
    if x == 0:
        x = 0x9E3779B97F4A7C15  # avoid zero lock
    while True:
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        y = (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
        yield y
```

## 概述

目标：在不使用实值权重或偏置、仅依赖连接的存在与符号（±1）、阈值与不应期的前提下，实现可扩展、稳定、可训练（无反向梯度）的脉冲神经网络。

关键思路：用距离驱动的随机连边形成小世界/层级模块化拓扑；用“生长‑剪枝‑稳态”的结构可塑性（STDP 式符号开关、老化剪枝、奖励门控）来学习；并通过入度恒定 + E/I 平衡 + 多延迟以及阈值/不应期来保证稳定与可分性。

新增：提供固定点训练模式（Fixed‑Point Mode），将推理、可塑性统计与采样统一为整数/定点运算（加/减/移位/查表/饱和），显著降低带宽与时延、提升可复现性与能效。

## 核心特性

- 仅符号突触：每条边只有符号 `a_ij ∈ {−1, +1}`（兴奋/抑制），无幅值、无偏置。
- 阈值 + 不应期：每个神经元使用固定阈值 `theta` 与不应期 `tau_ref` 作为稳定器与节流阀。
- 距离驱动连边：连边概率随几何/层间距离衰减，少量长程桥接边形成小世界拓扑。
- 从 0 连边冷启动：低率噪声或短时“老师驱动”点火，然后仅靠结构规则学习。
- 结构可塑性：在线增删边/翻转符号，维持入度近恒定与 E/I 平衡。
- 多延迟表达：用不同延迟的并行边表达时间模式（同符号同延迟合并，避免等效放大权重）。
- 事件驱动：稀疏邻接（CSR/CSC）配合事件队列，低发放率下计算/能耗近线性。
- 固定点模式（可选）：推理与可塑性统计使用整数/定点（Q 格式），提供量化标定、整数 STDP 与采样。

## 快速开始

- 环境：Python 3.11 + NumPy（纯 CPU）。
- 运行全部测试：`python -m unittest -v`（仓库根目录）。
- 运行指定测试：`python -m unittest tests/test_simulator.py -v`。
- 使用 `pytest`（可选）：`pytest -q` 或 `pytest tests/test_simulator.py -q`。

项目结构与导入（无需打包，直接从仓库根导入）：

- `core/` 事件驱动 LIF 模拟器与稀疏拓扑（CSR）
- `topology/` 拓扑构建器与生成器
- `plasticity/` 结构可塑性规则
- `monitor/` 运行监控与指标
- `readout/` 读出策略
- `config/` 配置与加载
- `tests/` 单元测试（unittest）

示例导入：`from core.simulator import EventDrivenLIF`

## 数学模型（离散 LIF + 不应期 + 延迟）

记膜电位 `v_i(t)`，脉冲指示 `s_i(t) ∈ {0,1}`，不应期计数 `ref_i(t)`，连接符号 `a_ij ∈ {−1,+1}`，传输延迟 `d_ij ≥ 1`，泄露系数 `lambda ∈ (0,1)`：

```
v_i(t+1) = v_reset                         if ref_i(t) > 0
         = lambda * v_i(t) + sum_j a_ij * s_j(t - d_ij)   if ref_i(t) = 0

s_i(t)   = 1 if v_i(t) >= theta else 0

ref_i(t+1) = tau_ref if s_i(t) = 1 else max(0, ref_i(t) - 1)

# 放电当步复位：若 s_i(t)=1，则 v_i(t+1) ← v_reset
```

## 拓扑与初始化（从 0 连边出发）

- 混合距离：`delta(i,j) = ||r_i - r_j||_2 + alpha * |ell_i - ell_j|`。
- 候选生长概率：`p_add(i→j) = kappa * exp(-delta/sigma) + rho_LR`（混入 0.5%–2% 长程）。
- 冷启动：从 0 连边开始；注入 0.1–1 Hz/神经元的低率泊松噪声点火。
- 再生长/剪枝：周期性按距离核补边、老化剪枝，维持目标入度 `K` 与 E/I 比约 1:1（±10%）。

## 无梯度学习：生长‑剪枝‑稳态

- STDP 式“符号开关”：基于时序相关统计在 {+1, −1} 间切换。
- 老化剪枝：长时间无效/低贡献的边移除；按距离核与约束再生长。
- 奖励门控：正确冻结/促兴奋，错误剪兴奋增抑制（可选）。
- 入度恒定：保持每个节点近似固定的入度 `K`，兼顾 E/I 平衡。
- 多延迟：保留不同延迟的并行边；同符号同延迟合并为一条。

## 阈值与不应期（σ 法则）

- 阈值：令输入噪声标准差为 `sigma_I`，取 `theta = beta * sigma_I`，`beta ∈ [1.5, 3]`。
- 不应期：`tau_ref = 2–5 ms` 起步，若出现同步爆发/短环自激，取 `5–10 ms`。
- 安全下界：`tau_ref ≥ d_min(E↔E)`（最短兴奋↔兴奋回路的传播延迟）。

## 互连与“多条连接”的安全约束

- 避免短延迟 E↔E 自激：若双向兴奋，确保最短延迟 ≥ `tau_ref`；或改为一侧抑制（满足 Dale）。
- 多条连接不等价放大：只保留不同延迟的并行边；同符号同延迟合并。
- 距离‑延迟一致：`d_ij ≈ round(||r_i - r_j|| / v)`，可加微小抖动。

## 读出与任务驱动（仍可无梯度）

- 无监督：时间窗内计数 → 聚类/原型最近邻；配侧抑制（WTA）。
- 监督（不反传）：读出层感知机规则（本地更新）；奖励门控结构学习。
- 全网保持完全无梯度。

## 规模化实现与资源估算

- 数据结构：CSR/CSC；建议每边约 8B（索引 4B + 符号 1B + 延迟 1B + 对齐）。
- 事件驱动成本：与活动突触数近线性；放电率越低越省。
- 内存示例：
  - `N=1e6, K=128 → M≈1.28e8` 边，约 0.95 GiB。
  - `N=1e5, K=64  → M≈6.4e6`  边，约 48.8 MiB。
- 并行：按空间 tile/模块切分；延迟队列批处理；跨边界事件队列交换。
- （可选）程序化连接：仅存节点种子与距离核，在放电时现场生成连接并合并事件，显著降低存储与 I/O（固定点模式尤佳）。

## 监控与调参闭环

- 平均放电率 `r_bar`：目标 1–10 Hz（任务相关）。
- 分枝因子 `b`：理想 0.9–1.1（近临界）。
- 互相关峰/雪崩尾：若异常，先增 `tau_ref`，再增 `theta`；必要时剪近程 E↔E、降长程比例。
- 调参顺序：先改结构（E/I、入度、短环、长程占比）与 `tau_ref`，后微调 `theta`。

## 固定点训练模式（推荐）

将推理/读出/结构可塑性/采样统一为整数/定点运算，替代浮点乘加：

- 数值格式建议：
  - 默认：`v, θ` 为 `int16`（Q4.11）；`λ` 为 `uint16`（Q1.15）；事件累加 `int32`；概率为 `uint16`（Q0.16）。
  - 不应期：`uint8` 步计数；读出/计数按规模选 `uint16` 或 `uint32`。
  - STDP 相关 `corr`：`int16/32` 饱和加 + 位移衰减；
  - 延迟 LUT：`uint8`（0–255 步）。
- 泄露实现：`v = ((lambda_q15 * v) >> 15) + I`；或 `v -= (v >> p) + I`（近似 `lambda ≈ 1 - 2^(-p)`）。
- 量化标定：估计 `W_eff ≈ 1/(1 - lambda)` 与输入方差，设定 `S_v` 与 `theta` 使非触发噪声下误触发率可控。
- 整数 STDP：位窗 + 饱和计分；周期性位移衰减；门控奖励。
- 采样：使用 Q0.16 概率与别名表（alias）实现 O(1) 采样。
- 读出：计数/最早放电时间的整数累计；可维护领先差 `Δ` 与未来可达上界 `U`，`Δ>U` 时提前判决。

## 参考伪代码

（A）固定点积分‑放电内核（C‑like，Q1.15 泄露）

```c
// v, theta: int16 (Q4.11)
// lambda_q15: uint16 (Q1.15)
// I_accum: 与 v 同量纲的 int32（已按 Q 缩放）
inline void integrate_and_fire_fp(
    int16_t* v, uint8_t* ref, const int32_t* I_accum,
    const int16_t theta, const uint16_t lambda_q15, int n, uint8_t tau_ref_steps)
{
    for (int i = 0; i < n; ++i) {
        if (ref[i]) { ref[i]--; continue; }

        int32_t vv = (int32_t)v[i] * (int32_t)lambda_q15; // Q4.11 * Q1.15 = Q5.26
        vv = (vv + (1<<14)) >> 15;                        // -> Q4.11（四舍五入）
        vv += I_accum[i];                                 // 累加事件

        if (vv >= theta) {        // 放电
            v[i] = 0;
            ref[i] = tau_ref_steps;
            // 记录 spike；加入 presyn 队列
        } else {
            if (vv >  32767) vv =  32767;                 // 饱和
            if (vv < -32768) vv = -32768;
            v[i] = (int16_t)vv;
        }
    }
}
```

（B）整数化 STDP 片段（位窗 + 饱和计分）

```c
// hist_* 为 W 位脉冲历史；corr 为 int32 饱和计数
corr += popcount(hist_pre  & (hist_post << delta)) * alpha_pos;
corr -= popcount(hist_post & (hist_pre  << delta)) * alpha_neg;
if ((step % decay_period) == 0) corr -= (corr >> decay_shift);
```

## 示例配置（YAML）

（仅作示意；实际实现不引入 YAML 依赖，可使用内置 JSON/最小配置解析）

```yaml
time:
  dt_ms: 1
  tau_m_ms: 50              # lambda = exp(-dt/tau_m) ≈ 0.98
  refractory_ms: 3

topology:
  K_in: 128
  EI_ratio: 1.0
  sigma_dist: 1.0
  alpha_layer: 2.0
  long_range_ratio: 0.01

threshold:
  beta_sigma: 2.0
  est_pre_rate_hz: 5

plasticity:
  prune_interval: 500
  prune_quota: 0.10
  stdp_window_ms: 20
  reward_modulated: false

stability_rules:
  merge_parallel_same_delay: true
  forbid_short_EE_loops: true

fixed_point:                # —— 固定点训练模式 ——
  enabled: true
  q_v_theta: "Q4.11"        # v 与 θ 的 Q 格式
  lambda_q15: true          # 使用 Q1.15 泄露；false 则用移位近似
  refractory_steps: 3
  i_accum_bits: 32          # I 累加器位宽
  corr_bits: 32             # STDP 相关累积位宽
  corr_decay_period: 512
  corr_decay_shift: 1       # 每个周期 corr -= corr >> 1
  alias_prob_q: "Q0.16"     # 概率/别名表定点格式
  tile_index_cap: 128       # （可选）程序化连接时每 tile 的索引上限

readout:
  window_steps: 200
  early_exit: false
```

## 常见问题（FAQ）

- 固定阈值会导致尺度不一致吗？
  - 通过入度近恒定、E/I 近平衡与泄露/不应期，大多数节点的输入尺度趋同；必要时可低频调整全局/分层尺度或在结构更新中做放电率门控剪/生。
- 从零连边如何点燃？
  - 使用 0.1–1 Hz/神经元的低率噪声或短时老师驱动；形成有用局部回路后可关闭或降低噪声。
- 出现同步爆发/雪崩怎么办？
  - 先把 `tau_ref` 提到 5–8 ms；剪短延迟 E↔E；提高抑制入度或降低长程比例；仍不稳再小幅增 `theta`。
- 固定点会影响精度吗？
  - 采用 Q4.11/Q3.12 + 饱和算术 + σ 法则标定，在事件合并/低发放率场景下通常与浮点效果接近；若不够，提升位宽或减小缩放。
  - 读出必须无梯度吗？
  - 是的；若允许，可在只读出层用感知机规则（仍无反传）提升样本效率。

## 快速开始（Quickstart）

下面示例构建一个小规模网络并运行 1000 步，打印每步读出与实时指标（预算占用、延后事件等）。

命令行运行

```
python scripts/run_local.py --steps 1000 --config examples/minimal.yml
```

其中 `examples/minimal.yml` 可为如下最小配置（支持 JSON 或最小 YAML）：

```yaml
time:
  dt_ms: 1
  tau_m_ms: 50
  refractory_ms: 3
topology:
  K_in: 64
  EI_ratio: 1.0
  long_range_ratio: 0.1
readout:
  window_steps: 200
```

脚本将打印如下字段：

- step: 步号（从 0 开始）
- spikes: 当步触发的脉冲数
- used_budget / deferred: 已用预算 / 延后事件数量
- label / latency: 读出预测标签与首达延迟（若未锁存则为 -1）

也可在不提供配置文件时直接运行，脚本会使用内置默认参数与随机输入样本。

### 配置字段一览（摘选）

- realtime
  - budget_per_step: 每步预算（整型，事件处理成本上限）
  - early_exit: 是否启用读出提前终止
- fixed_point
  - refractory_steps: LIF 不应期步数
- plasticity
  - period: 可塑性周期（步）
  - lr_num, lr_den: 别名重权学习率（整数分数）
  - corr_decay_period, corr_decay_shift: 相关度衰减配置（按周期 corr -= corr >> shift）
  - low_contrib_frac: 以全局发放计数选出底部比例的“低贡献”神经元（0..1）
  - reseed_rate: 对低贡献集合中按该概率重置“探索”通道的随机种子（0..1）
- connectivity
  - core_ratio, explore_ratio: 核心/探索配额比例
  - core_long_range_ratio, explore_long_range_ratio: 长程偏置比例（分配到最后四分之一 tiles）
  - near_radius: 近邻半径（tiles）；near_wrap: 是否采用环绕距离
  - ei_mapping_mode: E/I 划分模式（half/alternating/custom）；ei_tiles: 自定义 E tiles 列表
  - preaggregate: 是否启用运行器 pending 事件预聚合（按 (post_tile, delay) 聚合，减少 push 数量）；默认 false，建议在事件量大或预算紧张时开启
  - preaggregate_min_events: 触发预聚合的最小 pending 事件数阈值（默认 64）

### 探索种子重置（Reseed）

- 目的：在不改变“核心”通道确定性的前提下，周期性地为低贡献神经元的“探索”通道刷新随机种子，以增加拓扑探索多样性。
- 实现：`reseed_small_fraction(seeds_core:uint64[], seeds_flex:uint64[], low_contrib_mask:bool[], rate:float, epoch:int)`
  - 仅对 `low_contrib_mask=True` 的索引进行候选筛选；核心 `seeds_core` 永不改变。
  - 选择规则：对每个候选 i 计算 64 位混合哈希 `h_i = hash64(seeds_core[i] ^ seeds_flex[i] ^ mix(i, epoch))`，若 `h_i < rate * 2^64` 则选中。
  - 更新：对选中的探索种子执行按位异或 `seeds_flex[i] ^= hash64(epoch)`；其余不变。
  - 完全确定性：同样的 `(seeds_core, seeds_flex, low_contrib_mask, rate, epoch)` 组合得到一致结果。
- 关键开关：核心/探索配额由连接生成器 `core_ratio/explore_ratio` 控制；reseed 仅作用于“探索”配额对应的种子。

### 保存/恢复训练断点（Runner 快照）

- 保存当前状态（v/ref、seeds_core/seeds_flex、alias_corr_prob、corr_accum、tile 位窗、spike_counts、step_index、global_seed、alias_version、config 等）

```
python scripts/run_local.py --steps 500 --save-prefix checkpoints/run1
```

会生成 `checkpoints/run1_runner.bin`（二进制 + JSON 头）。

- 从快照继续运行，并可再次保存到新前缀：

```
python scripts/run_local.py --steps 500 --load-prefix checkpoints/run1 --save-prefix checkpoints/run1_next
```

说明
- 快照 I/O 使用 memmap + JSON meta，保持 dtype/shape 一致与跨平台确定性。
- 加载后会重建内部别名缓存，延迟到首次使用时按最新状态懒重建。
- stability_rules
  - forbid_short_EE_loops: 禁止短 E→E 回路
  - min_ee_delay: 短回路最小延迟
  - drop_short_EE: 短回路事件直接丢弃（否则提升到最小延迟）


## 默认超参速查表

- 时间步 `Δt`：1 ms；膜常数 `tau_m`：20–50 ms（`lambda = exp(−Δt/tau_m)`）。
- 入度 `K`：64–256；E/I 比：≈ 1:1（±10%）。
- 长程比例：0.5%–2%。
- 阈值：`theta ≈ beta * sigma_I`，`beta ∈ [1.5, 3]`。
- 不应期 `tau_ref`：2–5 ms（抑爆取 5–10 ms）。
- STDP 窗口：10–20 ms；剪枝配额：5%–15%/周期（结合入度门控）。
- 固定点 Q(v,θ)：Q4.11（或 Q3.12）；`lambda` 用 Q1.15 或移位近似；I 累加器：int32；概率：Q0.16。

## 项目结构与开发

- 新版包路径：`symbolicplastic_snn/`（逐步填充；与 `tests/` 一一对应）。
- 导入示例：`from symbolicplastic_snn.core.lif_fixedpoint import ...`。
- 测试：推荐 `pytest -q`（也兼容 `python -m unittest -v` 视测试风格而定）。
- 配置：不引入外部依赖，使用内置 JSON/最小配置解析；YAML 示例仅为文档说明。

## 许可证与免责声明

本仓库强调工程可执行性与可扩展性。不同硬件/任务下最佳参数可能不同；请结合“监控与调参闭环”中的指标进行闭环调优。若未在仓库另行声明，推荐使用通用开源许可证（如 MIT/Apache‑2.0）。

## 一句话总结

在“仅符号突触 + 阈值 + 不应期”的约束下，距离驱动的小世界拓扑配合无梯度的结构可塑性，即可支撑大规模、稳定、可学习的 SNN。开启固定点训练模式（Q 格式、整数 STDP/采样、饱和算术），按“σ 法则”设定 `theta`、以 `tau_ref` 抑爆，再以“生长‑剪枝‑稳态”在线更新结构，即可把系统跑稳、跑快、跑大。
