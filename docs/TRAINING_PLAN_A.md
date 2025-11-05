# 方案 A 启动指南（稳妥大规模）

适配本机（Apple M2 Max / 32 GiB）建议的稳妥训练规模与启动参数，确保步级事件数与内存占用处于可控范围，并保持确定性。

## 参数摘要
- 规模: `N = n_tiles * tile_size = 64 * 32768 = 2,097,152`
- 事件采样: `indices_per_event = 8`
- 实时预算: `budget_per_step = 160_000`
- 时间轮: `slots = 16`，内存上限 `bytes_cap = 2 GiB`
- 读出: `readout_window = 100`
- 神经元参数: `refractory_steps = 2`，`theta` 使用默认
- 输入发放率: `rate_max = 0.002`
- 连接采样配额: `core_ratio = 8/64 ≈ 0.125`，`explore_ratio = 1/64 ≈ 0.015625`
- 稳定存储上限: `promote_per_pre_cap = 32`

## 启动脚本
使用 `scripts/start_plan_a.py` 封装运行。核心逻辑：构造 `RunnerConfig`，设置时间轮 `bytes_cap`，用确定性 `FloatRng` 生成 `[0,1)` 输入并循环若干步。

示例：
- 运行 2,000 步、默认种子：
  - `python scripts/start_plan_a.py --steps 2000`
- 指定种子与时间轮上限：
  - `python scripts/start_plan_a.py --steps 5000 --seed 1234 --bytes-cap 2147483648`

脚本会周期性打印关键指标（预算使用率、延迟事件等），运行结束后输出规模 `N` 与总步数确认。

## 说明
- 大规模下，时间轮内存按组聚合自动降级；`bytes_cap` 可调以平衡细粒度聚合与内存占用。
- 若需进一步放大 `N`，建议同比降低 `indices_per_event` 或 `rate_max`，并检查 `budget_per_step` 与每步事件数的一致性。
- 全部随机性统一使用 `symbolicplastic_snn.core.prng` 的 `SeedSpace/Stream/FloatRng`，保证跨平台确定性。
