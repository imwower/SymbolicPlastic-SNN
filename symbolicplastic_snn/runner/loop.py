from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from symbolicplastic_snn.conn.alias import AliasForTile, build_alias
from symbolicplastic_snn.conn.generator import gen_block_events
from symbolicplastic_snn.core.lif_fixedpoint import ConfigFp, lif_step
from symbolicplastic_snn.encode.input_encoders import rate_encode_q016
from symbolicplastic_snn.realtime.budgeter import EventBudget
from symbolicplastic_snn.schedule.timewheel import BlockEvent, TimeWheel
from symbolicplastic_snn.readout.readout import Readout, ReadoutConfig
from symbolicplastic_snn.plasticity.stats import BitWindow, update_corr
from symbolicplastic_snn.plasticity.update import reweight_alias_smallstep, reseed_small_fraction
from symbolicplastic_snn.io.snapshot import save_snapshot, load_snapshot


@dataclass
class RunnerConfig:
    n_tiles: int = 4
    tile_size: int = 16
    indices_per_event: int = 8
    slots: int = 8
    readout_window: int = 50
    rate_max: float = 0.2
    theta: int = 4  # int16 threshold in Q domain used by LIF
    refractory_steps: int = 2
    budget_per_step: int = 1024
    early_exit: bool = False
    # Plasticity params
    plasticity_period: int = 50
    stdp_delta: int = 1
    corr_alpha_pos: int = 2
    corr_alpha_neg: int = 1
    lr_num: int = 1
    lr_den: int = 64
    low_contrib_frac: float = 0.2
    reseed_rate: float = 0.25
    # Core vs Explore quotas for generator
    core_ratio: float = 0.8
    explore_ratio: float = 0.2
    # Global seed controls all PRNGs (SplitMix64-based)
    # Note: `seed` param in constructor still used to init this.
    # Stability rules
    stability_forbid_short_EE_loops: bool = False
    stability_min_ee_delay: int = 2
    stability_drop_short_EE: bool = False
    # Long-range bias for core/explore alias (fraction of mass to last quarter tiles)
    core_long_range_ratio: float = 0.05
    explore_long_range_ratio: float = 0.5
    # Neighborhood radius (tiles) for core bias and priority P1
    near_radius: int = 1
    near_wrap: bool = True
    # E/I mapping for stability rules: 'half' | 'alternating' | 'custom'
    ei_mapping_mode: str = "half"
    ei_tiles: Optional[list[int]] = None


class SnnRunner:
    def __init__(self, cfg: RunnerConfig | None = None, seed: int = 1234) -> None:
        self.cfg = cfg or RunnerConfig()
        self.n_tiles = int(self.cfg.n_tiles)
        self.tile_size = int(self.cfg.tile_size)
        self.N = self.n_tiles * self.tile_size

        # PRNG seed
        self._global_seed = np.uint64(seed)

        # State
        self.v = np.zeros(self.N, dtype=np.int16)
        self.ref = np.zeros(self.N, dtype=np.uint8)

        # Alias distribution influenced by learned corr; start uniform
        base = 65535 // self.n_tiles
        w = np.full(self.n_tiles, base, dtype=np.uint16)
        rem = 65535 - base * self.n_tiles
        if rem > 0:
            w[:rem] = (w[:rem].astype(np.uint32) + 1).astype(np.uint16)
        self.alias_corr_prob = w  # uint16, sum=65535
        self._alias_version = 0
        self._alias_cache_core: dict[int, tuple[int, AliasForTile]] = {}
        self._alias_cache_explore: dict[int, tuple[int, AliasForTile]] = {}

        # Delay LUT (uint8), zeros by default
        self.delay_lut = np.zeros((self.n_tiles, self.n_tiles), dtype=np.uint8)

        # Seeds: core and flex, derived deterministically via SplitMix64 hashing
        self.seeds_core, self.seeds_flex = self._init_seeds_splitmix64()

        # Time wheel
        self.wheel = TimeWheel(slots=int(self.cfg.slots), bytes_cap=None)

        # Readout: two channels over halves by default
        self.readout = Readout(ReadoutConfig(window=self.cfg.readout_window, early_exit=bool(self.cfg.early_exit)))
        ids0 = np.arange(0, self.N // 2, dtype=np.int32)
        ids1 = np.arange(self.N // 2, self.N, dtype=np.int32)
        self.readout.add_channel("C0", ids0)
        self.readout.add_channel("C1", ids1)

        # LIF config and parameters
        self.lif_cfg = ConfigFp(refractory_steps=int(self.cfg.refractory_steps))
        self.lambda_q15 = np.uint16(32768)  # ~1.0 decay
        self.theta = np.int16(self.cfg.theta)

        self.step_durations: List[float] = []
        self._step_index = 0
        self.last_metrics: Dict[str, Any] = {}

        # Plasticity state
        self.tile_bw = BitWindow(width=16)
        self.corr_accum = np.zeros(self.n_tiles, dtype=np.int32)
        self.spike_counts = np.zeros(self.N, dtype=np.int32)
        # Reuse buffer for current I
        self._I_buf = np.zeros(self.N, dtype=np.int32)
        # Tile E/I mapping for stability rules
        self._tile_is_E = self._build_ei_mapping()

    # ---------------- Runner API ----------------
    def step(self, x_t: np.ndarray) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()
        # 1) Encode input into immediate events (delay=0) using integer Bernoulli with Q0.16 probabilities
        mask = self._encode_input_q016(np.asarray(x_t, dtype=np.float32))
        self._push_input_spikes(mask)

        # 2) Pop current slot, aggregate to I vector, and LIF update
        events = self.wheel.pop()
        I = self._aggregate_events(events)
        spikes, _ = lif_step(self.v, self.ref, I, self.theta, self.lambda_q15, self.lif_cfg)
        # Update spike counts for reseed stats
        if spikes.size:
            self.spike_counts += spikes.astype(np.int32)

        # Update tile bit-window history
        tile_spk = spikes.reshape(self.n_tiles, self.tile_size).any(axis=1)
        self.tile_bw.push(tile_spk)

        # 3) Generate next-step connectivity-driven events (prioritized, budgeted)
        budget_total = int(self.cfg.budget_per_step)
        budget = EventBudget(b_step=budget_total)
        processed_by_pri = {0: 0, 1: 0, 2: 0}
        deferred_events: List[BlockEvent] = []

        if np.any(spikes):
            spk_idx = np.nonzero(spikes)[0]
            # Collect events along with their pre_tile for priority classification
            pending: List[Tuple[BlockEvent, int]] = []
            quota_core_tiles = 0
            quota_explore_tiles = 0
            emitted_core = 0
            emitted_explore = 0
            for pre_id in spk_idx.tolist():
                pre_tile = pre_id // self.tile_size
                # Normalize ratios
                cr = max(0.0, float(self.cfg.core_ratio))
                er = max(0.0, float(self.cfg.explore_ratio))
                if (cr + er) <= 0.0:
                    cr_n, er_n = 1.0, 0.0
                else:
                    s = cr + er
                    cr_n, er_n = cr / s, er / s
                T_total = int(self.n_tiles)
                T_core = int(round(T_total * cr_n))
                T_explore = max(0, T_total - T_core)
                quota_core_tiles += T_core
                quota_explore_tiles += T_explore

                if T_core > 0:
                    be_core = gen_block_events(
                        pre_id=int(pre_id),
                        step=self._step_index,
                        pre_tile=int(pre_tile),
                        alias_tbl=self._get_alias_for(pre_tile, mode="core"),
                        tile_size=self.tile_size,
                        T_tiles=T_core,
                        M=self.cfg.indices_per_event,
                        seed=np.uint64(self.seeds_core[pre_id]),
                        delay_lut=self.delay_lut,
                        budget=None,
                    )
                    emitted_core += len(be_core)
                    for ev in be_core:
                        # Stability: forbid short E->E loops by increasing delay
                        ev2 = self._apply_stability_rules(ev, pre_tile)
                        if ev2 is not None:
                            pending.append((ev2, pre_tile))

                if T_explore > 0:
                    be_explore = gen_block_events(
                        pre_id=int(pre_id),
                        step=self._step_index,
                        pre_tile=int(pre_tile),
                        alias_tbl=self._get_alias_for(pre_tile, mode="explore"),
                        tile_size=self.tile_size,
                        T_tiles=T_explore,
                        M=self.cfg.indices_per_event,
                        seed=np.uint64(self.seeds_flex[pre_id]),
                        delay_lut=self.delay_lut,
                        budget=None,
                    )
                    emitted_explore += len(be_explore)
                    for ev in be_explore:
                        ev2 = self._apply_stability_rules(ev, pre_tile)
                        if ev2 is not None:
                            pending.append((ev2, pre_tile))

            # Determine leading channel tiles for P0 classification
            leader_idx = int(np.argmax(self.readout._counts)) if self.readout._counts is not None else 0
            tiles_per_channel = self.N // 2 // self.tile_size  # tiles per readout channel (2 channels)
            leader_tiles = set(range(leader_idx * tiles_per_channel, (leader_idx + 1) * tiles_per_channel))

            def event_priority(ev: BlockEvent, pre_tile: int) -> int:
                if ev.post_tile in leader_tiles:
                    return 0  # P0: readout-related
                # P1: near neighbor in tile space (distance <= 1)
                d = abs(int(ev.post_tile) - int(pre_tile))
                if d <= 1:
                    return 1
                return 2

            # Vectorized priority computation
            if pending:
                ev_arr = [p[0] for p in pending]
                pre_arr = np.array([p[1] for p in pending], dtype=np.int32)
                post_arr = np.array([ev.post_tile for ev in ev_arr], dtype=np.int32)
                leader_mask = np.isin(post_arr, np.fromiter(leader_tiles, count=len(leader_tiles), dtype=np.int32)) if leader_tiles else np.zeros(len(ev_arr), dtype=bool)
                dist = np.abs(post_arr - pre_arr)
                if self.cfg.near_wrap:
                    n = self.n_tiles
                    dist = np.minimum(dist, n - dist)
                pri_arr = np.where(leader_mask, 0, np.where(dist <= int(self.cfg.near_radius), 1, 2)).astype(np.int32)
                order = np.argsort(pri_arr, kind="stable")
                pending = [(ev_arr[i], int(pre_arr[i])) for i in order]

            # Consume budget by priority order
            for ev, pre_tile in pending:
                cost = int(ev.indices.size)
                # pri already computed; recompute cheaply for metrics
                if ev.post_tile in leader_tiles:
                    pri = 0
                else:
                    d = abs(int(ev.post_tile) - int(pre_tile))
                    if self.cfg.near_wrap:
                        n = self.n_tiles
                        d = min(d, n - d)
                    pri = 1 if d <= int(self.cfg.near_radius) else 2
                if budget.allow(cost):
                    budget.charge(cost)
                    self.wheel.push(ev)
                    processed_by_pri[pri] += 1
                else:
                    deferred_events.append(ev)

        # 4) Update readout; early exit can short-circuit remaining priorities
        self.readout.step(spikes)
        out = None
        # After processing P0, check for early exit using remaining budget
        if self.readout.cfg.early_exit and self.readout.can_early_exit(budget_remaining=(budget_total - budget.used)):
            out = self.readout.emit()
            # Defer all unprocessed events
            if deferred_events:
                self.wheel.defer_to_future(deferred_events, extra_delay=1)
            # Skip P1/P2 handling (already deferred above)
        else:
            # Not early exit: defer leftover due to budget
            if deferred_events:
                self.wheel.defer_to_future(deferred_events, extra_delay=1)

        # 5) Periodic plasticity: STDP corr -> alias reweight, then reseed exploration
        reweighted = False
        reseeded_cnt = 0
        period = max(1, int(self.cfg.plasticity_period))
        if self._step_index > 0 and (self._step_index % period == 0):
            # Compute corr deltas for each tile w.r.t global pre history (OR over tiles)
            # Global pre history as OR of tile histories
            pre_hist_global = np.uint32(0)
            if self.tile_bw._hist is not None:
                for h in self.tile_bw._hist:
                    pre_hist_global = np.uint32(int(pre_hist_global) | int(h))

            delta_vec = np.zeros(self.n_tiles, dtype=np.int32)
            for t in range(self.n_tiles):
                post_hist = self.tile_bw.get_history(t)
                d = update_corr(
                    pre_hist_global,
                    post_hist,
                    delta=int(self.cfg.stdp_delta),
                    alpha_pos=np.uint8(self.cfg.corr_alpha_pos),
                    alpha_neg=np.uint8(self.cfg.corr_alpha_neg),
                )
                delta_vec[t] = int(d)

            self.corr_accum = (self.corr_accum + delta_vec).astype(np.int32)
            # Apply decay per README if configured
            decay_p = getattr(self.cfg, "corr_decay_period", 512)
            decay_s = getattr(self.cfg, "corr_decay_shift", 1)
            if decay_p > 0 and (self._step_index % int(decay_p) == 0):
                self.corr_accum = (self.corr_accum - (self.corr_accum >> int(decay_s))).astype(np.int32)

            # Reweight alias probabilities across tiles
            new_prob_q = reweight_alias_smallstep(
                prob_q016=self.alias_corr_prob,
                corr_int32=self.corr_accum,
                lr_num=int(self.cfg.lr_num),
                lr_den=int(self.cfg.lr_den),
                keep_sum=True,
                ei_quota=None,
                long_range_ratio=None,
            )
            self.alias_corr_prob = new_prob_q
            self._alias_version += 1
            # Invalidate caches
            self._alias_cache_core.clear()
            self._alias_cache_explore.clear()
            reweighted = True
            # Decay or reset corr after applying
            # Keep running corr, do not zero; leave decay to rule above

            # Reseed low-contribution neurons in exploration seeds only
            # Determine bottom fraction by spike_counts
            frac = min(1.0, max(0.0, float(self.cfg.low_contrib_frac)))
            if frac > 0.0:
                # Compute integer threshold by order statistic (avoid float quantile)
                cnts = self.spike_counts
                if cnts.size > 0:
                    k = int(round(frac * max(0, cnts.size - 1)))
                    thresh = int(np.sort(cnts)[k])
                else:
                    thresh = 0
                low_mask = cnts <= thresh
                before = self.seeds_flex.copy()
                reseed_small_fraction(
                    self.seeds_core,
                    self.seeds_flex,
                    low_mask,
                    rate=float(self.cfg.reseed_rate),
                    epoch=self._step_index,
                )
                reseeded_cnt = int(np.count_nonzero(self.seeds_flex != before))

        self._step_index += 1
        t1 = time.perf_counter()
        dur = t1 - t0
        self.step_durations.append(dur)
        # Record metrics
        self.last_metrics = {
            "step_time_sec": dur,
            "budget_used": int(budget.used),
            "budget_total": budget_total,
            "budget_used_ratio": (float(budget.used) / budget_total) if budget_total > 0 else 0.0,
            "deferred_count": len(deferred_events),
            "processed_by_priority": processed_by_pri,
            "reweighted": reweighted,
            "reseeded_count": reseeded_cnt,
            # Quota metrics (may be 0 if no spikes)
            "quota_core_tiles": locals().get("quota_core_tiles", 0),
            "quota_explore_tiles": locals().get("quota_explore_tiles", 0),
            "emitted_core_events": locals().get("emitted_core", 0),
            "emitted_explore_events": locals().get("emitted_explore", 0),
            "spikes_count": int(spikes.sum()) if spikes.size else 0,
        }
        return out

    # ---------------- PRNG helpers ----------------
    _SPLITMIX64_INC = np.uint64(0x9E3779B97F4A7C15)
    _SPLITMIX64_M1 = np.uint64(0xBF58476D1CE4E5B9)
    _SPLITMIX64_M2 = np.uint64(0x94D049BB133111EB)

    def _hash64_vec(self, x: np.ndarray) -> np.ndarray:
        z = (x + self._SPLITMIX64_INC).astype(np.uint64)
        z ^= (z >> np.uint64(30))
        z = (z * self._SPLITMIX64_M1).astype(np.uint64)
        z ^= (z >> np.uint64(27))
        z = (z * self._SPLITMIX64_M2).astype(np.uint64)
        z ^= (z >> np.uint64(31))
        return z.astype(np.uint64)

    def _rng_uint16_for_step(self, step: int, size: int) -> np.ndarray:
        # Vectorized SplitMix64 stream: state_i = seed + INC*(i+1)
        base = (self._global_seed ^ np.uint64(step) ^ np.uint64(0xD1342543DE82EF95)).astype(np.uint64)
        idx = np.arange(1, size + 1, dtype=np.uint64)
        state = (base + idx * self._SPLITMIX64_INC).astype(np.uint64)
        z = state.copy()
        z ^= (z >> np.uint64(30))
        z = (z * self._SPLITMIX64_M1).astype(np.uint64)
        z ^= (z >> np.uint64(27))
        z = (z * self._SPLITMIX64_M2).astype(np.uint64)
        z ^= (z >> np.uint64(31))
        return ((z >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint16)

    def _init_seeds_splitmix64(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(self.N, dtype=np.uint64)
        a = (self._global_seed ^ (idx * np.uint64(0xD1342543DE82EF95)) ^ np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        b = (self._global_seed ^ (idx * np.uint64(0x94D049BB133111EB)) ^ np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        seeds_core = self._hash64_vec(a)
        seeds_flex = self._hash64_vec(b)
        return seeds_core.astype(np.uint64), seeds_flex.astype(np.uint64)

    # ---------------- Input encoding helpers ----------------
    def _encode_input_q016(self, x_t: np.ndarray) -> np.ndarray:
        # Convert float input to Q0.16 probabilities, then integer Bernoulli via per-step SplitMix64 stream
        x = np.asarray(x_t, dtype=np.float32)
        # p_float = clip(x * rate_max, 0, 1)
        p = np.clip(x * float(self.cfg.rate_max), 0.0, 1.0)
        # Convert to Q0.16 with rounding
        p_q = np.minimum((p * 65535.0 + 0.5).astype(np.int64), 65535).astype(np.uint16)
        r = self._rng_uint16_for_step(self._step_index, self.N)
        mask = rate_encode_q016(p_q, lambda size: r[:size])
        return mask.reshape(-1)

    # ---------------- Alias helpers ----------------
    def _renorm_q016(self, x: np.ndarray) -> np.ndarray:
        x32 = np.maximum(0, x.astype(np.int64))
        total = int(x32.sum())
        n = x32.size
        if total <= 0:
            base = 65535 // n
            out = np.full(n, base, dtype=np.int32)
            rem = 65535 - base * n
            if rem > 0:
                out[:rem] += 1
            return out.astype(np.uint16)
        prod = x32 * 65535
        base = (prod // total).astype(np.int32)
        rem = (prod % total).astype(np.int64)
        out = base.copy()
        need = int(65535 - int(base.sum(dtype=np.int64)))
        if need > 0:
            if need < n:
                idx = np.argpartition(-rem, need - 1)[:need]
            else:
                idx = np.arange(n)
            sel = idx[np.argsort(-rem[idx], kind="mergesort")][:need]
            out[sel] += 1
        return out.astype(np.uint16)

    def _base_weights_for_pre_tile(self, pre_tile: int, long_range_ratio: float, radius: int = 1) -> np.ndarray:
        n = self.n_tiles
        lr_ratio = min(1.0, max(0.0, float(long_range_ratio)))
        total = 65535
        lr_mass = int(round(total * lr_ratio))
        nr_mass = total - lr_mass
        w = np.zeros(n, dtype=np.int32)
        # Near set indices (wrap-around)
        near = set()
        for d in range(-radius, radius + 1):
            near.add((pre_tile + d) % n)
        far = [i for i in range(n) if i not in near]
        # Distribute masses
        if len(near) > 0:
            base = nr_mass // len(near)
            for i in near:
                w[i] = base
            rem = nr_mass - base * len(near)
            for i in list(near)[:rem]:
                w[i] += 1
        if len(far) > 0:
            base = lr_mass // len(far)
            for i in far:
                w[i] += base
            rem = lr_mass - base * len(far)
            for i in far[:rem]:
                w[i] += 1
        return np.clip(w, 0, 65535).astype(np.uint16)

    def _get_alias_for(self, pre_tile: int, mode: str) -> AliasForTile:
        cache = self._alias_cache_core if mode == "core" else self._alias_cache_explore
        tmp = cache.get(pre_tile)
        if tmp is not None and tmp[0] == self._alias_version:
            return tmp[1]
        lr = self.cfg.core_long_range_ratio if mode == "core" else self.cfg.explore_long_range_ratio
        base = self._base_weights_for_pre_tile(pre_tile, lr, radius=int(self.cfg.near_radius))
        # Combine with learned corr
        prod = base.astype(np.int64) * self.alias_corr_prob.astype(np.int64)
        w = self._renorm_q016(prod)
        prob, alias = build_alias(w)
        af = AliasForTile(prob=prob, alias=alias)
        cache[pre_tile] = (self._alias_version, af)
        return af

    # ---------------- Stability rules ----------------
    def _apply_stability_rules(self, ev: BlockEvent, pre_tile: int) -> Optional[BlockEvent]:
        if not self.cfg.stability_forbid_short_EE_loops:
            return ev
        # Define E/I split by tiles: first half tiles are E
        tiles_per_chan = max(1, (self.N // 2) // self.tile_size)
        pre_E = pre_tile < tiles_per_chan
        post_E = int(ev.post_tile) < tiles_per_chan
        if pre_E and post_E and int(ev.delay) < int(self.cfg.stability_min_ee_delay):
            if self.cfg.stability_drop_short_EE:
                return None
            return BlockEvent(
                post_tile=int(ev.post_tile),
                indices=np.array(ev.indices, dtype=np.int32, copy=True),
                k=np.array(ev.k, dtype=np.int16, copy=True),
                delay=int(self.cfg.stability_min_ee_delay),
                capped=bool(ev.capped),
                total_k=np.int64(ev.total_k),
            )
        return ev

    # ---------------- Snapshot helpers ----------------
    def save_state(self, prefix: str) -> None:
        # Save core-related state
        core_alias = {"prob": self.alias_corr_prob, "alias": build_alias(self.alias_corr_prob)[1]}
        save_snapshot(f"{prefix}_core.bin", self.v, self.ref, self.seeds_core, core_alias, {"mode": "core"})
        explore_alias = {"prob": self.alias_corr_prob, "alias": build_alias(self.alias_corr_prob)[1]}
        save_snapshot(f"{prefix}_explore.bin", self.v, self.ref, self.seeds_flex, explore_alias, {"mode": "explore"})

    def load_state(self, prefix: str) -> None:
        v, ref, seeds_core, alias_core, _ = load_snapshot(f"{prefix}_core.bin")
        _, _, seeds_flex, alias_exp, _ = load_snapshot(f"{prefix}_explore.bin")
        self.v = v.astype(np.int16)
        self.ref = ref.astype(np.uint8)
        self.seeds_core = seeds_core.astype(np.uint64)
        self.seeds_flex = seeds_flex.astype(np.uint64)
        # Restore corr-driven alias prob from snapshot
        self.alias_corr_prob = alias_core["prob"].astype(np.uint16)
        self._alias_version += 1
        self._alias_cache_core.clear()
        self._alias_cache_explore.clear()

    def run(self, X_T: Iterable[np.ndarray]) -> Dict[str, Any]:
        out: Optional[Dict[str, Any]] = None
        for x_t in X_T:
            got = self.step(x_t)
            if got is not None:
                out = got
            # advance wheel pointer at the end of time step
            self.wheel.tick()

        # Final emit if nothing early-exited
        return out or self.readout.emit()

    # ---------------- Helpers ----------------
    def _push_input_spikes(self, mask: np.ndarray) -> None:
        # Group by tile and push immediate events with k=1 per index
        if mask.size != self.N:
            raise ValueError("input mask length mismatch")
        if not np.any(mask):
            return
        for t in range(self.n_tiles):
            start = t * self.tile_size
            end = start + self.tile_size
            idx_local = np.nonzero(mask[start:end])[0]
            if idx_local.size == 0:
                continue
            k = np.ones(idx_local.size, dtype=np.int16)
            ev = BlockEvent(post_tile=t, indices=idx_local.astype(np.int32), k=k, delay=0)
            self.wheel.push(ev)

    def _aggregate_events(self, events: List[BlockEvent]) -> np.ndarray:
        I = self._I_buf
        I.fill(0)
        if not events:
            return I
        # Handle capped events: accumulate per tile add
        tile_add = np.zeros(self.n_tiles, dtype=np.int32)
        noncapped = []
        for ev in events:
            if ev.capped:
                if ev.total_k > 0:
                    add = int(ev.total_k) // self.tile_size
                    if add > 0:
                        tile_add[int(ev.post_tile)] += add
            else:
                noncapped.append(ev)
        # Apply capped adds
        tiles = np.nonzero(tile_add > 0)[0]
        for t in tiles:
            s = t * self.tile_size
            e = s + self.tile_size
            I[s:e] += tile_add[t]
        # Flatten non-capped indices and accumulate via add.at
        if noncapped:
            total_len = int(sum(ev.indices.size for ev in noncapped))
            idx = np.empty(total_len, dtype=np.int32)
            val = np.empty(total_len, dtype=np.int32)
            off = 0
            for ev in noncapped:
                n = ev.indices.size
                if n == 0:
                    continue
                base = int(ev.post_tile) * self.tile_size
                idx[off:off+n] = base + ev.indices
                val[off:off+n] = ev.k.astype(np.int32)
                off += n
            if off > 0:
                np.add.at(I, idx[:off], val[:off])
        return I

    def _build_ei_mapping(self) -> np.ndarray:
        mode = (self.cfg.ei_mapping_mode or "half").lower()
        n = self.n_tiles
        if mode == "alternating":
            return (np.arange(n) % 2 == 0)
        if mode == "custom" and self.cfg.ei_tiles is not None:
            mask = np.zeros(n, dtype=bool)
            for t in self.cfg.ei_tiles:
                if 0 <= int(t) < n:
                    mask[int(t)] = True
            return mask
        # default 'half'
        half = n // 2
        mask = np.zeros(n, dtype=bool)
        mask[:half] = True
        return mask


__all__ = ["RunnerConfig", "SnnRunner"]
