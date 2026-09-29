#%%

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import permutations
from tqdm import tqdm
import numpy as np

import mean_field_module as mfm

STORE_FORMAT = "stepping_stone_window/v1"
C_KEY_SCALE = 10 ** 6  # committed fractions are memoised at 1e-6 granularity


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass
class DynamicsParams:
    """Everything that changes what a single evaluation *means*.

    These fields form the cache signature: change any of them and previously
    cached evaluations are kept but placed in a separate namespace, never reused.
    """
    TIME: int = 1000
    dt: float = 0.1
    # 'fixed_budget': the target phase always gets `target_steps` steps, so a
    #   failure can never be an artefact of a long intermediate phase eating the
    #   clock. This differs from the original script and is the default here.
    # 'fixed_total': target budget = TIME - t_inter (original behaviour).
    target_time_mode: str = "fixed_budget"
    target_steps: int | None = None  # None -> TIME
    success_threshold: float = 0.98
    success_window: int = 3
    stall_detection: bool = True
    stall_window: int = 50
    stall_tol: float = 1e-9

    def budget(self, t_inter: int) -> int:
        base = self.TIME if self.target_steps is None else int(self.target_steps)
        if self.target_time_mode == "fixed_total":
            return max(0, self.TIME - int(t_inter))
        return base

    def signature(self) -> dict:
        return asdict(self)


@dataclass
class SearchParams:
    """Everything that only changes *where* we look. Not part of the signature."""
    commit_resolution: float = 0.01
    time_resolution: float = 0.001
    tau_max: float = 1.0          # upper end of the tau grid (exclusive)
    probe_max_gap: float = 0.005  # dense sweep spacing as a fraction of TIME:
                                  # the window detection floor
    n_guards: int = 6             # cheap first-pass probes before the sweep
    flat_tol: float = 1e-6        # per-step L1 motion below which the state is settled
    flat_margin: float = 0.02     # safety margin after last motion, fraction of TIME
    n_flat_probes: int = 3        # probes spent on the settled tail
    probe_cap: int = 400          # ceiling on sweep probes; widens the spacing
                                  # if it binds, and the realised floor is reported
    c_coarse_stride: int = 5      # coarse c stride, in fine-grid index units
    profile_resolution: float | None = None  # stage-2 tau step; None -> time_resolution
    checkpoint_every: int = 25


# --------------------------------------------------------------------------- #
# model context
# --------------------------------------------------------------------------- #
class ModelContext:
    """Precomputed reduced-state quantities shared by every simulation."""

    def __init__(self, q_total, H: int):
        q_H, keys_H, steady_0, steady_1, steady_2 = mfm.H_reducer(q_total, H)
        F, Finv = mfm.integer_mapping(keys_H)
        self.choices, self.nn_choices = mfm.reverse_shift_vectorized(Finv)
        self.q_s = mfm.integer_probabilities(q_H, F)
        self.empty_state = F['']
        self.Ss = [F[steady_0], F[steady_1], F[steady_2]]
        self.P_key, self.P_value = mfm.state_transitions(q_H, H, F, Finv)
        self.H = H

    def make_llm(self, steady_state, signal, c, population, dt):
        return mfm.LLM_dynamics_integer(
            q_s=self.q_s,
            P_key=self.P_key,
            P_value=self.P_value,
            empty_state=self.empty_state,
            choices=self.choices,
            nn_choices=self.nn_choices,
            steady_state=steady_state,
            CM=c,
            committment_signal=signal,
            population=population,
            dt=dt,
        )


def _copy_pop(p):
    if isinstance(p, np.ndarray):
        return p.copy()
    return copy.deepcopy(p)


def _probs(llm) -> np.ndarray:
    return np.exp(np.asarray(llm.population_log_probs, dtype=float))


# --------------------------------------------------------------------------- #
# the two phases
# --------------------------------------------------------------------------- #
class IntermediateTrajectory:
    """The B-commitment phase for one c, simulated once and snapshotted.

    Every (c, t_inter) pair in the original script re-ran the intermediate phase
    from scratch. Here it is run once per c, monotonically forward, with the
    population cached at each visited step, so scanning tau costs only target
    phases. Snapshots are dropped after `snapshot_cap` entries, after which
    earlier states are recovered by replaying from the nearest snapshot.
    """

    def __init__(self, ctx: ModelContext, steady_state, intermediate: int,
                 c: float, dyn: DynamicsParams, snapshot_cap: int = 4096):
        self.ctx = ctx
        self.steady_state = steady_state
        self.intermediate = intermediate
        self.c = c
        self.dyn = dyn
        self.snapshot_cap = snapshot_cap

        self.llm = ctx.make_llm(steady_state, intermediate, c, None, dyn.dt)
        self.llm.initialize_population_steady()
        self.t = 0
        self.snapshots = {0: _copy_pop(self.llm.population)}
        self._buf = deque(maxlen=dyn.success_window)
        self.flip_time = None  # first step at which the population converged on B
        self._prev_p = _probs(self.llm)
        self.motion = []       # motion[k] = ||p(k+1) - p(k)||_1

    def _advance_to(self, t: int):
        while self.t < t:
            self.llm.algorithmic_update_asynchronous()
            self.t += 1
            p = _probs(self.llm)
            self.motion.append(float(np.abs(p - self._prev_p).sum()))
            self._prev_p = p
            self._buf.append(float(p[self.intermediate]))
            if (self.flip_time is None
                    and len(self._buf) == self._buf.maxlen
                    and float(np.mean(self._buf)) >= self.dyn.success_threshold):
                self.flip_time = self.t
            if len(self.snapshots) < self.snapshot_cap:
                self.snapshots[self.t] = _copy_pop(self.llm.population)

    def flat_time(self, t_max: int, tol: float, margin: int) -> int:
        """Last step at which the state still moved, plus a margin.

        Deliberately the *last* motion rather than the first quiet stretch: a
        slow early build-up must not be mistaken for a settled state, or early
        switching times would be under-probed.
        """
        self._advance_to(int(t_max))
        m = np.asarray(self.motion[:int(t_max)], dtype=float)
        moving = np.nonzero(m > tol)[0]
        last = int(moving[-1]) + 1 if moving.size else 0
        return int(min(int(t_max), last + int(margin)))

    def population_at(self, t: int):
        t = int(t)
        if t in self.snapshots:
            return _copy_pop(self.snapshots[t])
        if t > self.t:
            self._advance_to(t)
            return _copy_pop(self.snapshots.get(t, self.llm.population))
        base = max(k for k in self.snapshots if k <= t)
        llm = self.ctx.make_llm(self.steady_state, self.intermediate, self.c,
                                _copy_pop(self.snapshots[base]), self.dyn.dt)
        for _ in range(t - base):
            llm.algorithmic_update_asynchronous()
        return _copy_pop(llm.population)


def run_target_phase(ctx: ModelContext, population, steady_state, target: int,
                     c: float, max_steps: int, dyn: DynamicsParams) -> dict:
    """Commit c to `target` from `population`; stop as soon as A is reached.

    Returns the outcome *and the final adoption vector over all signals*, which
    is what distinguishes the failure modes beyond the window (intermediate
    entrenched vs. incumbent restored vs. unresolved).

    Note on `stopped_early`: successful runs are cut at the moment the criterion
    is met, so their `final_probs` is the state at that moment, not at the end of
    the budget. Failed runs always report the state at the end of the budget (or
    at the stall). Adoption values beyond the window are therefore terminal
    states; adoption values inside it are threshold-crossing states.
    """
    if max_steps <= 0:
        return {"success": False, "t_success": None, "reason": "no_budget",
                "steps": 0, "final_probs": None, "stopped_early": False}

    llm = ctx.make_llm(steady_state, target, c, _copy_pop(population), dyn.dt)
    buf = deque(maxlen=dyn.success_window)
    prev = None
    still = 0
    p = None

    for step in range(1, int(max_steps) + 1):
        llm.algorithmic_update_asynchronous()
        p = _probs(llm)
        buf.append(float(p[target]))
        if len(buf) == buf.maxlen and float(np.mean(buf)) >= dyn.success_threshold:
            return {"success": True, "t_success": step, "reason": "converged",
                    "steps": step, "final_probs": p.tolist(), "stopped_early": True}
        if dyn.stall_detection:
            if prev is not None and np.nanmax(np.abs(p - prev)) <= dyn.stall_tol:
                still += 1
                if still >= dyn.stall_window:
                    return {"success": False, "t_success": None, "reason": "stalled",
                            "steps": step, "final_probs": p.tolist(),
                            "stopped_early": True}
            else:
                still = 0
            prev = p
    return {"success": False, "t_success": None, "reason": "budget",
            "steps": int(max_steps),
            "final_probs": None if p is None else p.tolist(),
            "stopped_early": False}


# --------------------------------------------------------------------------- #
# memoised evaluator
# --------------------------------------------------------------------------- #
def c_key(c: float) -> int:
    return int(round(float(c) * C_KEY_SCALE))


class PairEvaluator:
    """Evaluates and caches success(c, t_inter) for one (S, B, A) triple."""

    def __init__(self, ctx: ModelContext, s_idx: int, target: int, intermediate: int,
                 dyn: DynamicsParams, ns: dict, save_cb=None, checkpoint_every: int = 25):
        self.ctx = ctx
        self.s_idx = s_idx
        self.target = target
        self.intermediate = intermediate
        self.dyn = dyn
        self.ns = ns
        self.save_cb = save_cb
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.n_new = 0
        self.n_hits = 0
        self.n_dense_probes = 0
        self.max_gap_used = 0
        self.last_schedule = None
        self._traj = None
        self._traj_key = None

    # -- intermediate trajectory (one live object at a time) ----------------- #
    def trajectory(self, c: float) -> IntermediateTrajectory:
        k = c_key(c)
        if self._traj_key != k:
            self._traj = IntermediateTrajectory(
                self.ctx, self.ctx.Ss[self.s_idx], self.intermediate, c, self.dyn)
            self._traj_key = k
        return self._traj

    def release(self):
        """Drop snapshots for the current c (called when the c-scan moves on)."""
        self._traj = None
        self._traj_key = None

    # -- evaluation ---------------------------------------------------------- #
    def evaluate(self, c: float, t_inter: int, need_probs: bool = False) -> dict:
        key = (self.s_idx, self.target, c_key(c), int(t_inter))
        cached = self.ns["evals"].get(key)
        if cached is not None and not (need_probs and cached.get("final_probs") is None
                                       and cached.get("reason") != "no_budget"):
            self.n_hits += 1
            return cached

        traj = self.trajectory(c)
        pop = traj.population_at(int(t_inter))
        out = run_target_phase(self.ctx, pop, self.ctx.Ss[self.s_idx], self.target,
                               c, self.dyn.budget(int(t_inter)), self.dyn)
        out = {
            "success": bool(out["success"]),
            "t_success": out["t_success"],
            "reason": out["reason"],
            "budget": self.dyn.budget(int(t_inter)),
            "final_probs": out["final_probs"],
            "stopped_early": out["stopped_early"],
        }
        self.ns["evals"][key] = out
        self.ns["flipB"][(self.s_idx, self.target, c_key(c))] = traj.flip_time
        self.n_new += 1
        if self.save_cb is not None and self.n_new % self.checkpoint_every == 0:
            self.save_cb()
        return out

    def success(self, c: float, t_inter: int) -> bool:
        return self.evaluate(c, t_inter)["success"]


# --------------------------------------------------------------------------- #
# search primitives
# --------------------------------------------------------------------------- #
def fine_t_grid(dyn: DynamicsParams, sp: SearchParams) -> np.ndarray:
    """Intermediate-phase durations, in integration steps (tau = t/TIME)."""
    step = max(1, int(round(sp.time_resolution * dyn.TIME)))
    t_max = int(round(sp.tau_max * dyn.TIME))
    return np.arange(0, max(t_max, 1), step, dtype=int)


def fine_c_grid(sp: SearchParams, c_max: float) -> np.ndarray:
    res = sp.commit_resolution
    n = int(np.floor((c_max - 1e-12) / res)) + 1
    g = np.round(np.arange(max(n, 1)) * res, 10)
    return g[(g < c_max - 1e-12) & (g > 0)]  # c = 0 cannot flip anything


def probe_any_tau(ev: "PairEvaluator", c: float, tgrid: np.ndarray,
                  sp: SearchParams, skip_zero: bool = True):
    """Stage-1 question: does *some* switching time work at this c?

    Two passes over [0, T_flat]; switching times past flattening are all the
    same state, so a few tail probes cover them.

      1. `n_guards` evenly spaced guard probes. A success here answers the
         question immediately -- the cheap path that keeps the outer c scan
         affordable.
      2. Failing that, a dense sweep at a fixed spacing of
         `probe_max_gap * TIME` steps. Every window at least that wide contains
         a probe, whatever the terminal regimes do: the sweep navigates by
         nothing, so multiple windows, disordered regime sequences and windows
         buried inside a uniform regime are all covered.

    T_flat is defined by motion alone, not by which convention leads, so it is
    valid whether the trajectory settles on the intermediate, on a mixed state,
    or barely leaves the incumbent.
    """
    traj = ev.trajectory(c)
    n = len(tgrid)
    spacing = int(tgrid[1] - tgrid[0]) if n > 1 else 1
    t_flat = traj.flat_time(int(tgrid[-1]), sp.flat_tol,
                            int(round(sp.flat_margin * ev.dyn.TIME)))
    n_active = max(1, min(n, int(np.searchsorted(tgrid, t_flat, side="right"))))

    lo_i = 1 if (skip_zero and n and tgrid[0] == 0) else 0
    hi_i = n_active - 1

    # -- pass 1: guards, plus a few probes on the settled tail --------------- #
    guards = [int(round(g)) for g in
              np.linspace(lo_i, max(lo_i, hi_i), max(2, sp.n_guards))]
    if n_active < n and sp.n_flat_probes > 0:
        guards += [int(round(g)) for g in
                   np.linspace(n_active - 1, n - 1, sp.n_flat_probes + 1)[1:]]
    guards = sorted(set(i for i in guards if lo_i <= i < n))

    for i in guards:
        if ev.evaluate(c, int(tgrid[i]), need_probs=True)["success"]:
            ev.last_schedule = _sched_meta(t_flat, spacing, spacing, len(guards),
                                           0, ev.dyn.TIME)
            return int(i)

    # -- pass 2: dense sweep at the declared floor --------------------------- #
    stride = max(1, int(round((sp.probe_max_gap * ev.dyn.TIME) / spacing)))
    n_sweep = len(range(lo_i, hi_i + 1, stride))
    if sp.probe_cap and n_sweep > sp.probe_cap:      # widen, and report honestly
        stride = int(np.ceil((hi_i - lo_i + 1) / sp.probe_cap))
    sweep = [i for i in range(lo_i, hi_i + 1, stride)]
    if sweep and sweep[-1] != hi_i:
        sweep.append(hi_i)

    n_dense = 0
    for i in sweep:
        if i in guards:
            continue
        n_dense += 1
        if ev.evaluate(c, int(tgrid[i]), need_probs=True)["success"]:
            ev.last_schedule = _sched_meta(t_flat, stride * spacing, spacing,
                                           len(guards), n_dense, ev.dyn.TIME)
            return int(i)

    ev.n_dense_probes += n_dense
    ev.max_gap_used = max(ev.max_gap_used, stride * spacing)
    ev.last_schedule = _sched_meta(t_flat, stride * spacing, spacing,
                                   len(guards), n_dense, ev.dyn.TIME)
    return None


def _sched_meta(t_flat, floor_steps, spacing, n_guards, n_dense, TIME) -> dict:
    """`floor_steps` is the realised sweep spacing: windows narrower than this,
    inside [0, T_flat], can be missed. A null result is exact to this width."""
    return {"t_flat": int(t_flat),
            "detection_floor_steps": int(floor_steps),
            "detection_floor_tau": float(floor_steps / TIME),
            "grid_step": int(spacing),
            "n_guards": int(n_guards), "n_dense_probes": int(n_dense)}


def profile_t_grid(dyn: DynamicsParams, sp: SearchParams) -> np.ndarray:
    """Stage-2 switching-time grid (may be coarser than the stage-1 fine grid)."""
    res = sp.time_resolution if sp.profile_resolution is None else sp.profile_resolution
    step = max(1, int(round(res * dyn.TIME)))
    t_max = int(round(sp.tau_max * dyn.TIME))
    return np.arange(0, max(t_max, 1), step, dtype=int)


def _runs(mask: np.ndarray) -> list:
    """Maximal runs of True, as inclusive index pairs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _terminal_label(out: dict, s_idx: int, intermediate: int, target: int,
                    threshold: float) -> str:
    """Where the population ended up: which convention, or 'unresolved'."""
    fp = out.get("final_probs")
    if fp is None:
        return "unknown"
    fp = np.asarray(fp, dtype=float)
    k = int(np.argmax(fp))
    if fp[k] < threshold:
        return "unresolved"
    return {s_idx: "incumbent", intermediate: "intermediate",
            target: "target"}.get(k, f"signal_{k}")


def _modal_regime(labels: list) -> dict:
    if not labels:
        return {"regime": None, "fraction": None, "n": 0}
    vals, counts = np.unique(np.array(labels, dtype=object).astype(str),
                             return_counts=True)
    k = int(np.argmax(counts))
    return {"regime": str(vals[k]), "fraction": float(counts[k] / len(labels)),
            "n": len(labels)}


def profile_at_c_star(ev: "PairEvaluator", c: float, tgrid: np.ndarray,
                      dyn: DynamicsParams, s_idx: int, intermediate: int,
                      target: int) -> dict:
    """STAGE 2. Dense sweep of switching times at c*, recording final adoption.

    Switching times are visited in ascending order so the intermediate
    trajectory advances forward once and every state is taken from a snapshot.
    Cached evaluations make an interrupted profile resume for free. The window
    edges are read off this profile by thresholding, so contiguity is observed
    rather than assumed.
    """
    n = len(tgrid)
    p_target = np.full(n, np.nan)
    p_inter = np.full(n, np.nan)
    p_incumbent = np.full(n, np.nan)
    success = np.zeros(n, dtype=bool)
    t_success = np.full(n, -1, dtype=int)
    stopped_early = np.zeros(n, dtype=bool)
    terminal, reason = [], []

    for i, t in tqdm(enumerate(tgrid)):
        out = ev.evaluate(c, int(t), need_probs=True)
        success[i] = out["success"]
        t_success[i] = -1 if out["t_success"] is None else int(out["t_success"])
        stopped_early[i] = bool(out.get("stopped_early", False))
        reason.append(out["reason"])
        fp = out.get("final_probs")
        if fp is None:
            terminal.append("unknown")
            continue
        fp = np.asarray(fp, dtype=float)
        p_target[i] = fp[target]
        p_inter[i] = fp[intermediate]
        p_incumbent[i] = fp[s_idx]
        terminal.append(_terminal_label(out, s_idx, intermediate, target,
                                        dyn.success_threshold))

    spacing = int(tgrid[1] - tgrid[0]) if n > 1 else 1
    nonzero = tgrid > 0
    segs_idx = _runs(success & nonzero)
    segments = [{
        "t_lo": int(tgrid[a]), "t_hi": int(tgrid[b]),
        "t_width": int(tgrid[b] - tgrid[a] + spacing),
        "tau_lo": float(tgrid[a] / dyn.TIME), "tau_hi": float(tgrid[b] / dyn.TIME),
        "tau_width": float((tgrid[b] - tgrid[a] + spacing) / dyn.TIME),
    } for a, b in segs_idx]

    out = {
        "found": bool(segments),
        "n_windows": len(segments),
        "segments": segments,
        "t_spacing": spacing,
        "tau0_control_success": bool(success[0]) if n and tgrid[0] == 0 else None,
        "profile": {
            "t_inter": tgrid.copy(),
            "tau": tgrid / dyn.TIME,
            "p_target": p_target,
            "p_intermediate": p_inter,
            "p_incumbent": p_incumbent,
            "success": success,
            "t_success": t_success,
            "stopped_early": stopped_early,
            "terminal": terminal,
            "reason": reason,
            "role_order": ["incumbent", "intermediate", "target"],
        },
    }
    if not segments:
        out["all_outcomes_regime"] = _modal_regime(
            [terminal[i] for i in range(n) if nonzero[i]])
        return out

    k = int(np.argmax([sg["t_width"] for sg in segments]))
    a, b = segs_idx[k]
    out.update({
        "t_lo": segments[k]["t_lo"], "t_hi": segments[k]["t_hi"],
        "t_width": segments[k]["t_width"],
        "tau_lo": segments[k]["tau_lo"], "tau_hi": segments[k]["tau_hi"],
        "tau_width": segments[k]["tau_width"],
        "primary_segment_index": k,
        "first_window_t_lo": segments[0]["t_lo"],
        "contiguous": len(segments) == 1,
        "left_edge_censored": bool(tgrid[a] <= spacing),
        "right_edge_censored": bool(b == n - 1),
        "pre_window_regime": _modal_regime(
            [terminal[i] for i in range(a) if nonzero[i]]),
        "post_window_regime": _modal_regime(terminal[b + 1:]),
        "p_target_in_window_min": float(np.nanmin(p_target[a:b + 1])),
        "p_target_post_window_mean": (float(np.nanmean(p_target[b + 1:]))
                                      if b + 1 < n else None),
        "p_intermediate_post_window_mean": (float(np.nanmean(p_inter[b + 1:]))
                                            if b + 1 < n else None),
    })
    return out


def find_c_star(ev: PairEvaluator, cgrid: np.ndarray, tgrid: np.ndarray,
                sp: SearchParams):
    """Smallest c on the fine grid for which some probed tau > 0 works.

    Coarse ascending scan to bracket the threshold, then bisection inside the
    bracket. Bisection assumes success is monotone in c *within one coarse
    bracket only*; the assumption is recorded in the result so it can be
    re-tested with `--c-mode scan` (exhaustive ascending fine scan, no
    monotonicity assumed, considerably more expensive).
    """
    n = len(cgrid)
    if n == 0:
        return None, {"bracket": None, "monotonicity_assumed": False}

    coarse = list(range(0, n, max(1, sp.c_coarse_stride)))
    if coarse[-1] != n - 1:
        coarse.append(n - 1)

    prev_fail, hit = None, None
    for j in coarse:
        ev.release()
        if probe_any_tau(ev, float(cgrid[j]), tgrid, sp) is not None:
            hit = j
            break
        prev_fail = j

    if hit is None:
        return None, {"bracket": None, "monotonicity_assumed": False}
    if prev_fail is None:
        return int(hit), {"bracket": [None, float(cgrid[hit])],
                          "monotonicity_assumed": False}

    lo, hi = prev_fail, hit
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ev.release()
        if probe_any_tau(ev, float(cgrid[mid]), tgrid, sp) is not None:
            hi = mid
        else:
            lo = mid
    return int(hi), {"bracket": [float(cgrid[prev_fail]), float(cgrid[hit])],
                     "monotonicity_assumed": True}


def find_c_star_scan(ev: PairEvaluator, cgrid: np.ndarray, tgrid: np.ndarray,
                     sp: SearchParams):
    for j in range(len(cgrid)):
        ev.release()
        if probe_any_tau(ev, float(cgrid[j]), tgrid, sp) is not None:
            return int(j), {"bracket": None, "monotonicity_assumed": False}
    return None, {"bracket": None, "monotonicity_assumed": False}


# --------------------------------------------------------------------------- #
# per-pair driver
# --------------------------------------------------------------------------- #
def analyse_pair(ctx: ModelContext, ns: dict, s_idx: int, target: int,
                 cm_dict: dict, dyn: DynamicsParams, sp: SearchParams,
                 save_cb=None, c_mode: str = "bisect", verbose: bool = True) -> dict:
    intermediate = [i for i in range(len(ctx.Ss)) if i not in (s_idx, target)][0]

    direct = (cm_dict.get(s_idx, {}) or {}).get(target)
    inter_info = (cm_dict.get(s_idx, {}) or {}).get(intermediate)
    direct_cm = direct["critical_mass"] if direct else None
    direct_cm_missing = direct_cm is None
    if direct_cm_missing:
        direct_cm = 1.0

    res = {
        "initial": s_idx,
        "intermediate": intermediate,
        "target": target,
        "direct_critical_mass": None if direct_cm_missing else float(direct_cm),
        "direct_critical_mass_missing": bool(direct_cm_missing),
        "direct_transition_time": direct.get("time") if direct else None,
        "intermediate_critical_mass": (inter_info or {}).get("critical_mass"),
        "intermediate_transition_time": (inter_info or {}).get("time"),
        "window_exists": False,
        "c_star": None,
        "status": "no_window",
    }

    cgrid = fine_c_grid(sp, float(direct_cm))
    tgrid = fine_t_grid(dyn, sp)
    if len(cgrid) == 0:
        res["status"] = "no_room_below_direct_cm"
        return res

    ev = PairEvaluator(ctx, s_idx, target, intermediate, dyn, ns,
                       save_cb=save_cb, checkpoint_every=sp.checkpoint_every)
    t0 = time.time()

    if c_mode == "scan":
        j, cinfo = find_c_star_scan(ev, cgrid, tgrid, sp)
    else:
        j, cinfo = find_c_star(ev, cgrid, tgrid, sp)

    if j is None:
        res.update({
            "status": "no_window",
            "search": _search_meta(ev, cgrid, tgrid, sp, dyn, cinfo, c_mode, t0),
        })
        if verbose:
            print(f"  [{s_idx}->{intermediate}->{target}] no window below "
                  f"c={direct_cm:.3f} ({ev.n_new} new evals)")
        return res

    c_star = float(cgrid[j])
    ev.release()
    pgrid = profile_t_grid(dyn, sp)
    win = profile_at_c_star(ev, c_star, pgrid, dyn, s_idx, intermediate, target)
    control = ev.evaluate(c_star, 0, need_probs=True)  # direct transition at c*

    res.update({
        "window_exists": bool(win.get("found", False)),
        "c_star": c_star,
        "c_star_index": int(j),
        "absolute_gain": (None if direct_cm_missing else float(direct_cm) - c_star),
        "relative_gain": (None if direct_cm_missing or direct_cm == 0
                          else (float(direct_cm) - c_star) / float(direct_cm)),
        "direct_success_at_c_star": bool(control["success"]),
        "control_inconsistent": bool(control["success"]),
        "intermediate_flip_time_at_c_star":
            ns["flipB"].get((s_idx, target, c_key(c_star))),
        "flat_time_at_c_star": (None if ev.last_schedule is None
                                else ev.last_schedule["t_flat"]),
        "status": "complete" if win.get("found") else "c_star_without_window",
    })
    res.update({k: v for k, v in win.items() if k != "found"})
    res["search"] = _search_meta(ev, cgrid, tgrid, sp, dyn, cinfo, c_mode, t0)

    if verbose:
        print(f"  [{s_idx}->{intermediate}->{target}] c*={c_star:.3f} "
              f"(direct {direct_cm:.3f}), tau in "
              f"[{res.get('tau_lo')}, {res.get('tau_hi')}], "
              f"{ev.n_new} new / {ev.n_hits} cached evals")
    return res


def _search_meta(ev, cgrid, tgrid, sp, dyn, cinfo, c_mode, t0) -> dict:
    return {
        "c_mode": c_mode,
        "c_bracket": cinfo.get("bracket"),
        "monotonicity_in_c_assumed": cinfo.get("monotonicity_assumed"),
        "commit_resolution": sp.commit_resolution,
        "time_resolution": sp.time_resolution,
        "profile_resolution": (sp.time_resolution if sp.profile_resolution is None
                               else sp.profile_resolution),
        "tau_max": sp.tau_max,
        "n_c_grid": int(len(cgrid)),
        "n_t_grid": int(len(tgrid)),
        "probe_max_gap_requested": sp.probe_max_gap,
        "n_guards": sp.n_guards,
        "detection_floor_steps": int(ev.max_gap_used),
        "stage1_mode": "guards+dense_sweep",
        "detection_floor_tau": float(ev.max_gap_used / dyn.TIME) if dyn.TIME else None,
        "probe_schedule_at_last_c": ev.last_schedule,
        "n_dense_probes": int(ev.n_dense_probes),
        "n_new_evaluations": int(ev.n_new),
        "n_cached_evaluations": int(ev.n_hits),
        "wall_seconds": round(time.time() - t0, 1),
        "dynamics": dyn.signature(),
    }


# --------------------------------------------------------------------------- #
# persistent store
# --------------------------------------------------------------------------- #
def sig_hash(dyn: DynamicsParams) -> str:
    blob = json.dumps(dyn.signature(), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def load_store(fname: str) -> dict:
    try:
        with open(fname, "rb") as f:
            store = pickle.load(f)
        if store.get("format") != STORE_FORMAT:
            raise ValueError(f"unrecognised store format in {fname}")
        return store
    except FileNotFoundError:
        return {"format": STORE_FORMAT, "namespaces": {}}


def get_namespace(store: dict, dyn: DynamicsParams) -> dict:
    h = sig_hash(dyn)
    return store["namespaces"].setdefault(
        h, {"signature": dyn.signature(), "evals": {}, "flipB": {}, "results": {}})


def save_store(store: dict, fname: str):
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    tmp = f"{fname}.tmp_{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp, fname)


# --------------------------------------------------------------------------- #
# file plumbing (unchanged conventions from the original experiment script)
# --------------------------------------------------------------------------- #
def find_matrix_file(names, pattern: str) -> str:
    for n1, n2, n3 in permutations(names, 3):
        fn = pattern.format(name1=n1, name2=n2, name3=n3)
        if os.path.exists(fn):
            return fn
    n1, n2, n3 = names
    return pattern.format(name1=n1, name2=n2, name3=n3)


def get_critical_mass_data(data_dict: dict) -> dict:
    return {
        s: {
            sig: ({"critical_mass": data_dict[s][sig]["solution"]["critical_mass"],
                   "time": data_dict[s][sig]["solution"]["time"]}
                  if "solution" in data_dict[s][sig] else None)
            for sig in data_dict[s]
        }
        for s in data_dict
    }


def run_single_option(args) -> str:
    options, shorthand, H, dyn, sp, c_mode, pairs, force = args
    try:
        q_pattern = (f"policies/log_q_dicts/Q_dict_{shorthand}"
                     "_{name1}_{name2}_{name3}_" + "0.5tmp.pkl")
        with open(find_matrix_file(options, q_pattern), "rb") as f:
            q_total = pickle.load(f)

        cm_pattern = (f"meta_data_async/mean_field_critical_mass/LLM_dynamics_{shorthand}"
                      "_{name1}_{name2}_{name3}_" + f"{H}mem_0.5tmp.pkl")
        with open(find_matrix_file(options, cm_pattern), "rb") as f:
            cm_dict = get_critical_mass_data(pickle.load(f))

        out_pattern = (f"meta_data_async/mean_field_window_search/WINDOW_{shorthand}"
                       "_{name1}_{name2}_{name3}_" + f"{H}mem_0.5tmp.pkl")
        out_file = find_matrix_file(options, out_pattern)

        store = load_store(out_file)
        ns = get_namespace(store, dyn)
        store.setdefault("options", list(options))
        store.setdefault("shorthand", shorthand)

        ctx = ModelContext(q_total, H)
        save_cb = lambda: save_store(store, out_file)

        print(f"[pid {os.getpid()}] {options} -> {out_file}")
        for s_idx in range(len(ctx.Ss)):
            for target in range(len(ctx.Ss)):
                if target == s_idx:
                    continue
                if pairs and (s_idx, target) not in pairs:
                    continue
                if not force and ns["results"].get((s_idx, target), {}).get("status") in (
                        "complete", "no_window", "no_room_below_direct_cm"):
                    print(f"  [{s_idx}->{target}] already done, skipping")
                    continue
                ns["results"][(s_idx, target)] = analyse_pair(
                    ctx, ns, s_idx, target, cm_dict, dyn, sp,
                    save_cb=save_cb, c_mode=c_mode)
                save_store(store, out_file)

        save_store(store, out_file)
        return f"OK {options}"
    except Exception as exc:  # keep one bad triple from killing the pool
        import traceback
        traceback.print_exc()
        return f"FAIL {options}: {exc}"


# --------------------------------------------------------------------------- #
# summary table
# --------------------------------------------------------------------------- #
def summary_records(store: dict) -> list:
    """Flatten a store into DataFrame-ready rows (one per S->B->A triple)."""
    rows = []
    for h, ns in store.get("namespaces", {}).items():
        for (s_idx, target), r in ns.get("results", {}).items():
            rows.append({
                "signature": h,
                "options": store.get("options"),
                "shorthand": store.get("shorthand"),
                "initial": s_idx,
                "intermediate": r.get("intermediate"),
                "target": target,
                "initial_name": (store.get("options") or [None] * 3)[s_idx]
                                if store.get("options") else None,
                "intermediate_name": (store.get("options") or [None] * 3)[r["intermediate"]]
                                     if store.get("options") and r.get("intermediate") is not None else None,
                "target_name": (store.get("options") or [None] * 3)[target]
                               if store.get("options") else None,
                "direct_cm": r.get("direct_critical_mass"),
                "intermediate_cm": r.get("intermediate_critical_mass"),
                "window_exists": r.get("window_exists"),
                "c_star": r.get("c_star"),
                "absolute_gain": r.get("absolute_gain"),
                "relative_gain": r.get("relative_gain"),
                "tau_lo": r.get("tau_lo"),
                "tau_hi": r.get("tau_hi"),
                "tau_width": r.get("tau_width"),
                "t_lo": r.get("t_lo"),
                "t_hi": r.get("t_hi"),
                "t_width": r.get("t_width"),
                "n_windows": r.get("n_windows"),
                "contiguous": r.get("contiguous"),
                "pre_window_regime": (r.get("pre_window_regime") or {}).get("regime"),
                "post_window_regime": (r.get("post_window_regime") or {}).get("regime"),
                "post_window_regime_fraction":
                    (r.get("post_window_regime") or {}).get("fraction"),
                "p_target_post_window_mean": r.get("p_target_post_window_mean"),
                "p_intermediate_post_window_mean":
                    r.get("p_intermediate_post_window_mean"),
                "no_window_regime": (r.get("all_outcomes_regime") or {}).get("regime"),
                "left_edge_censored": r.get("left_edge_censored"),
                "right_edge_censored": r.get("right_edge_censored"),
                "flip_B_time_at_c_star": r.get("intermediate_flip_time_at_c_star"),
                "flat_time_at_c_star": r.get("flat_time_at_c_star"),
                "detection_floor_tau": (r.get("search") or {}).get("detection_floor_tau"),
                "n_dense_probes": (r.get("search") or {}).get("n_dense_probes"),
                "control_inconsistent": r.get("control_inconsistent"),
                "status": r.get("status"),
            })
    return rows


def build_summary(directory: str, out_csv: str | None = None) -> list:
    rows = []
    for fn in sorted(os.listdir(directory)):
        if not fn.startswith("WINDOW_") or not fn.endswith(".pkl"):
            continue
        rows.extend(summary_records(load_store(os.path.join(directory, fn))))
    if out_csv:
        import csv
        if rows:
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        print(f"wrote {len(rows)} rows to {out_csv}")
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="search for stepping-stone windows")
    r.add_argument("--shorthand", default="qwen25_7B")
    r.add_argument("--H", type=int, default=3)
    r.add_argument("--options-file", default="policies/emotion_triplets.pkl")
    r.add_argument("--processes", type=int, default=10)
    r.add_argument("--time", type=int, default=1000, dest="TIME")
    r.add_argument("--dt", type=float, default=0.1)
    r.add_argument("--target-time-mode", default="fixed_total",
                   choices=["fixed_budget", "fixed_total"])
    r.add_argument("--target-steps", type=int, default=None)
    r.add_argument("--commit-resolution", type=float, default=0.01)
    r.add_argument("--time-resolution", type=float, default=0.001)
    r.add_argument("--tau-max", type=float, default=1.0)
    r.add_argument("--probe-max-gap", type=float, default=0.005,
                   help="dense-sweep spacing as a fraction of TIME; this is "
                        "the window detection floor")
    r.add_argument("--n-guards", type=int, default=6,
                   help="cheap first-pass probes per candidate c before the dense sweep")
    r.add_argument("--flat-tol", type=float, default=1e-6,
                   help="per-step L1 motion below which the state counts as settled")
    r.add_argument("--flat-margin", type=float, default=0.02,
                   help="safety margin after last motion, as a fraction of TIME")
    r.add_argument("--n-flat-probes", type=int, default=3)
    r.add_argument("--probe-cap", type=int, default=400,
                   help="ceiling on dense-sweep probes per candidate c")
    r.add_argument("--c-coarse-stride", type=int, default=5)
    r.add_argument("--profile-resolution", type=float, default=None,
                   help="stage-2 tau step at c* (default: --time-resolution)")
    r.add_argument("--c-mode", default="bisect", choices=["bisect", "scan"])
    r.add_argument("--no-stall", action="store_true")
    r.add_argument("--pairs", default=None,
                   help="restrict to pairs, e.g. '2:0,1:0' (initial:target)")
    r.add_argument("--force", action="store_true",
                   help="recompute finished pairs (cached evaluations are still reused)")

    s = sub.add_parser("summary", help="flatten stores into a table")
    s.add_argument("--dir", default="meta_data_async/mean_field_window_search")
    s.add_argument("--csv", default=None)

    a = p.parse_args(argv)

    if a.cmd == "summary":
        rows = build_summary(a.dir, a.csv)
        for row in rows:
            print(row)
        return

    dyn = DynamicsParams(
        TIME=a.TIME, dt=a.dt, target_time_mode=a.target_time_mode,
        target_steps=a.target_steps, stall_detection=not a.no_stall)
    sp = SearchParams(
        commit_resolution=a.commit_resolution, time_resolution=a.time_resolution,
        tau_max=a.tau_max, probe_max_gap=a.probe_max_gap, n_guards=a.n_guards,
        flat_tol=a.flat_tol, flat_margin=a.flat_margin,
        n_flat_probes=a.n_flat_probes, probe_cap=a.probe_cap,
        c_coarse_stride=a.c_coarse_stride, profile_resolution=a.profile_resolution)

    pairs = set()
    if a.pairs:
        for tok in a.pairs.split(","):
            i, j = tok.split(":")
            pairs.add((int(i), int(j)))

    with open(a.options_file, "rb") as f:
        all_options = pickle.load(f)

    args = [(opts, a.shorthand, a.H, dyn, sp, a.c_mode, pairs, a.force)
            for opts in all_options]
    n_proc = max(1, min(a.processes, len(args)))
    print(f"searching {len(args)} triplets on {n_proc} processes "
          f"(signature {sig_hash(dyn)})")

    if n_proc == 1:
        results = [run_single_option(x) for x in args]
    else:
        with mp.Pool(processes=n_proc) as pool:
            results = pool.map(run_single_option, args)

    print("\n" + "=" * 50)
    for res in results:
        print(res)


if __name__ == "__main__":
    main()

# %%