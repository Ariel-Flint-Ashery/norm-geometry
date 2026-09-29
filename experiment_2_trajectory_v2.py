#%%
"""
stepping_stone_trajectory.py
============================

Single-triple companion to `window_search_v2.py`, in two modes.

run   For one model, one (source S, intermediate B, target A) triple and one
      committed fraction c:
        (i)  optionally, the full trajectory at one switching time tau, with
             `population_log_probs` recorded after every integration step;
        (ii) a sweep over switching times t_inter = 0, 1, ..., TIME - 1
             (stride configurable). For each t_inter, the final probabilities
             of every action are averaged over the last `--avg-window` (= 3)
             steps of the simulation.
plot  Nature-style single-column figure: (a) the trajectory at tau,
      (b) final adoption against switching time, with the window of effective
      switching shaded.

Protocol
--------
    phase 0  the minority c holds B for t_inter steps, starting from the
             steady state of S;
    phase 1  the same minority switches to A for DynamicsParams.budget(t_inter)
             steps. Under the default `fixed_total` mode every run ends at
             absolute step TIME, so all switching times share one horizon.

Snapshot trick (sweep)
----------------------
The intermediate phase is independent of t_inter, so it is simulated once, up
to the largest t_inter, with the population stored every `--snapshot-every`
steps (default 1). Each switching time then starts the target phase from its
snapshot (replaying at most `snapshot_every - 1` intermediate steps), which
reduces the cost from O(TIME^2) intermediate + target steps to O(TIME) + target
steps. Target phases are independent and can run on several processes.

The last-`avg-window` average is taken over the concatenated trajectory, so
when the target budget is shorter than the window the average includes the
final intermediate steps.

Consistency with the window search
----------------------------------
Trajectories are built exactly as in `PairEvaluator.evaluate` (ModelContext,
`initialize_population_steady()` on an LLM committing to B, population
hand-over into a fresh LLM committing to A). `--check-cache` compares the
tau trajectory with the cached window-search outcome, provided the dynamics
flags match those of the `run` that produced the cache.
Deliberate differences: target phases run their full budget (no early stop,
no stall detection; `--no-stall` only matches the cache signature).

Signal indexing (VERIFY)
------------------------
Signal ids are indices into `ctx.Ss`. This script assumes the order equals the
name order of the Q-dict file found on disk; `--signal-order` overrides it.

Usage
-----
    python stepping_stone_trajectory.py run  --source sadness \\
        --intermediate surprise --target fear --c 0.15 --tau 0.06 --processes 8
    python stepping_stone_trajectory.py plot --source sadness \\
        --intermediate surprise --target fear --c 0.15 --tau 0.06 \\
        --x-max 250 --name-roles
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import time
from collections import deque
from itertools import permutations

import numpy as np
from tqdm import tqdm

import window_search_v2 as ws

TRAJ_FORMAT = "stepping_stone_trajectory/v1"
SWEEP_FORMAT = "stepping_stone_sweep/v1"

Q_PATTERN = "policies/log_q_dicts/Q_dict_{shorthand}_{{name1}}_{{name2}}_{{name3}}_0.5tmp.pkl"
CM_PATTERN = ("meta_data_async/mean_field_critical_mass/LLM_dynamics_{shorthand}"
              "_{{name1}}_{{name2}}_{{name3}}_{H}mem_0.5tmp.pkl")
WINDOW_PATTERN = ("meta_data_async/mean_field_window_search/WINDOW_{shorthand}"
                  "_{{name1}}_{{name2}}_{{name3}}_{H}mem_0.5tmp.pkl")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_file(names, pattern: str):
    """First permutation of `names` that exists on disk, and that permutation."""
    for perm in permutations(names, 3):
        fn = pattern.format(name1=perm[0], name2=perm[1], name3=perm[2])
        if os.path.exists(fn):
            return fn, list(perm)
    return None, None


def _logp(llm) -> np.ndarray:
    return np.array(llm.population_log_probs, dtype=float, copy=True)


def _converged(buf: deque, threshold: float) -> bool:
    return len(buf) == buf.maxlen and float(np.mean(buf)) >= threshold


def _first_convergence(probs: np.ndarray, idx: int, window: int, thr: float):
    """First step t >= window at which mean(probs[t-window+1:t+1, idx]) >= thr.

    Row 0 is the initial state and is excluded, as in the window search.
    """
    x = probs[1:, idx]
    if len(x) < window:
        return None
    m = np.convolve(x, np.ones(window) / window, mode="valid")
    hit = np.nonzero(m >= thr)[0]
    return None if hit.size == 0 else int(hit[0] + window)


# --------------------------------------------------------------------------- #
# single trajectory
# --------------------------------------------------------------------------- #
def simulate(ctx: ws.ModelContext, s_idx: int, b_idx: int, a_idx: int, c: float,
             t_inter: int, dyn: ws.DynamicsParams, stop_on_success: bool = False,
             progress: bool = True) -> dict:
    """S --(c on B, t_inter steps)--> hand-over --(c on A, budget steps)-->."""
    t_inter = int(t_inter)
    budget = int(dyn.budget(t_inter))
    steady = ctx.Ss[s_idx]
    thr = dyn.success_threshold
    bar = tqdm(total=t_inter + budget, disable=not progress,
               desc=f"trajectory t_inter={t_inter}")

    rows, phase = [], []

    llm = ctx.make_llm(steady, b_idx, c, None, dyn.dt)
    llm.initialize_population_steady()
    rows.append(_logp(llm))
    phase.append(0)
    for _ in range(t_inter):
        llm.algorithmic_update_asynchronous()
        rows.append(_logp(llm))
        phase.append(0)
        bar.update(1)

    llm_a = ctx.make_llm(steady, a_idx, c, ws._copy_pop(llm.population), dyn.dt)
    handover_gap = None
    try:  # diagnostic only: does the fresh object report the same log-probs?
        handover_gap = float(np.nanmax(np.abs(_logp(llm_a) - rows[-1])))
    except Exception:
        pass

    buf_a = deque(maxlen=dyn.success_window)
    t_succ = None
    for k in range(1, budget + 1):
        llm_a.algorithmic_update_asynchronous()
        lp = _logp(llm_a)
        rows.append(lp)
        phase.append(1)
        buf_a.append(float(np.exp(lp[a_idx])))
        bar.update(1)
        if t_succ is None and _converged(buf_a, thr):
            t_succ = k
            if stop_on_success:
                break
    bar.close()

    log_probs = np.vstack(rows)
    phase = np.asarray(phase, dtype=np.int8)
    step = np.arange(log_probs.shape[0])
    probs = np.exp(log_probs)
    final_probs = probs[-1]
    return {
        "step": step,
        "tau": step / dyn.TIME,
        "phase": phase,
        "commit_signal": np.where(phase == 0, b_idx, a_idx).astype(np.int8),
        "log_probs": log_probs,
        "t_inter": t_inter,
        "tau_inter": t_inter / dyn.TIME,
        "budget": budget,
        "n_target_steps_run": int(step[-1] - t_inter),
        "stopped_early": bool(stop_on_success and t_succ is not None
                              and t_succ < budget),
        "flip_time_intermediate": _first_convergence(
            probs[:t_inter + 1], b_idx, dyn.success_window, thr),
        "success": t_succ is not None,
        "t_success": t_succ,                       # steps after the switch
        "t_success_abs": None if t_succ is None else t_inter + t_succ,
        "final_probs": final_probs,
        "terminal": ws._terminal_label({"final_probs": final_probs.tolist()},
                                       s_idx, b_idx, a_idx, thr),
        "handover_max_abs_logp_gap": handover_gap,
    }


def check_against_cache(res: dict, store_path, dyn, s_idx, a_idx, c) -> dict:
    """Compare success and t_success with the window-search memo, if present."""
    if store_path is None or not os.path.exists(store_path):
        return {"status": "no_store"}
    store = ws.load_store(store_path)
    ns = store["namespaces"].get(ws.sig_hash(dyn))
    if ns is None:
        return {"status": "no_namespace", "signature": ws.sig_hash(dyn),
                "available": list(store["namespaces"])}
    ev = ns["evals"].get((s_idx, a_idx, ws.c_key(c), int(res["t_inter"])))
    if ev is None:
        return {"status": "not_evaluated"}
    agree = (bool(ev["success"]) == res["success"]
             and (not ev["success"] or ev["t_success"] == res["t_success"]))
    return {"status": "agree" if agree else "DISAGREE", "cached": ev}


# --------------------------------------------------------------------------- #
# switching-time sweep
# --------------------------------------------------------------------------- #
def intermediate_snapshots(ctx, s_idx, b_idx, c, t_top, dyn, every=1,
                           progress=True):
    """Run phase 0 once to `t_top`; store populations every `every` steps."""
    llm = ctx.make_llm(ctx.Ss[s_idx], b_idx, c, None, dyn.dt)
    llm.initialize_population_steady()
    snaps = {0: ws._copy_pop(llm.population)}
    rows = [_logp(llm)]
    for t in tqdm(range(1, t_top + 1), disable=not progress,
                  desc="intermediate phase"):
        llm.algorithmic_update_asynchronous()
        rows.append(_logp(llm))
        if t % every == 0:
            snaps[t] = ws._copy_pop(llm.population)
    return snaps, np.vstack(rows)


_G: dict = {}  # per-process state: model context and intermediate probs


def _init_worker(q_file, H, inter_probs, seed):
    if seed is not None:
        np.random.seed(seed + os.getpid())
        random.seed(seed + os.getpid())
    with open(q_file, "rb") as f:
        _G["ctx"] = ws.ModelContext(pickle.load(f), H)
    _G["inter_probs"] = inter_probs


def _sweep_chunk(job):
    """Target phases for a chunk of switching times.

    job = (items, s_idx, b_idx, a_idx, c, dyn, window), with
    items = [(t_inter, base_t, base_population), ...].
    """
    items, s_idx, b_idx, a_idx, c, dyn, window = job
    ctx, inter_probs = _G["ctx"], _G["inter_probs"]
    steady = ctx.Ss[s_idx]
    out = []
    for t, base_t, base_pop in items:
        pop = ws._copy_pop(base_pop)
        if t > base_t:                               # replay from the snapshot
            llm_b = ctx.make_llm(steady, b_idx, c, pop, dyn.dt)
            for _ in range(t - base_t):
                llm_b.algorithmic_update_asynchronous()
            pop = ws._copy_pop(llm_b.population)

        tail = deque(inter_probs[max(0, t - window + 1):t + 1], maxlen=window)
        last_lp = np.log(inter_probs[t])
        buf = deque(maxlen=dyn.success_window)
        t_succ = None
        llm = ctx.make_llm(steady, a_idx, c, pop, dyn.dt)
        for k in range(1, int(dyn.budget(t)) + 1):
            llm.algorithmic_update_asynchronous()
            last_lp = _logp(llm)
            p = np.exp(last_lp)
            tail.append(p)
            buf.append(float(p[a_idx]))
            if t_succ is None and _converged(buf, dyn.success_threshold):
                t_succ = k
        out.append((t, np.mean(np.vstack(tail), axis=0), last_lp, t_succ))
    return out


def run_sweep(q_file, H, ctx, s_idx, b_idx, a_idx, c, dyn, t_grid, window,
              every=1, processes=1, chunk=10, seed=None, done=None,
              save_cb=None, progress=True):
    """Return {t_inter: (mean_probs, final_log_probs, t_success)}."""
    done = {} if done is None else dict(done)
    todo = [int(t) for t in t_grid if int(t) not in done]
    t_top = int(max(t_grid))
    snaps, inter_lp = intermediate_snapshots(ctx, s_idx, b_idx, c, t_top, dyn,
                                             every, progress)
    inter_probs = np.exp(inter_lp)

    jobs = []
    for i in range(0, len(todo), chunk):
        items = []
        for t in todo[i:i + chunk]:
            base = (t // every) * every
            items.append((t, base, snaps[base]))
        jobs.append((items, s_idx, b_idx, a_idx, c, dyn, window))

    bar = tqdm(total=len(todo), disable=not progress, desc="target phases")
    n_done_chunks = 0

    def _collect(res):
        nonlocal n_done_chunks
        for t, m, lp, ts in res:
            done[t] = (m, lp, ts)
        bar.update(len(res))
        n_done_chunks += 1
        if save_cb is not None and n_done_chunks % 10 == 0:
            save_cb(done, inter_lp)

    if processes <= 1:
        _G["ctx"], _G["inter_probs"] = ctx, inter_probs
        for job in jobs:
            _collect(_sweep_chunk(job))
    else:
        with mp.Pool(processes, initializer=_init_worker,
                     initargs=(q_file, H, inter_probs, seed)) as pool:
            for res in pool.imap_unordered(_sweep_chunk, jobs):
                _collect(res)
    bar.close()
    return done, inter_lp


def assemble_sweep(done, t_grid, inter_lp, a_idx, b_idx, dyn, window):
    t_grid = np.asarray(sorted(int(t) for t in t_grid))
    ok = np.array([t in done for t in t_grid])
    t_ok = t_grid[ok]
    mean_probs = np.vstack([done[t][0] for t in t_ok])
    final_lp = np.vstack([done[t][1] for t in t_ok])
    t_succ = np.array([-1 if done[t][2] is None else done[t][2] for t in t_ok])
    return {
        "t_inter": t_ok,
        "tau": t_ok / dyn.TIME,
        "budget": np.array([dyn.budget(int(t)) for t in t_ok]),
        "mean_final_probs": mean_probs,          # (n, K), last-`window` average
        "final_log_probs": final_lp,             # (n, K), last step only
        "t_success": t_succ,                     # first crossing after switch, -1 if none
        "success_end": mean_probs[:, a_idx] >= dyn.success_threshold,
        "avg_window": int(window),
        "intermediate_log_probs": inter_lp,      # phase 0 run to max(t_inter)
        "flip_time_intermediate": _first_convergence(
            np.exp(inter_lp), b_idx, dyn.success_window, dyn.success_threshold),
        "complete": bool(ok.all()),
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
MM = 1.0 / 25.4
SINGLE_COL_W = 89 * MM
BASE_FS = 7
PANEL_FS = 8

ROLE_STYLE = {  # Okabe-Ito; distinct dashes for greyscale
    "Source":       dict(color="#CC79A7", ls=(0, (4, 1.6)),         lw=1.1, z=5),
    "Intermediate": dict(color="#E69F00", ls=(0, (4, 1.2, 1, 1.2)), lw=1.1, z=6),
    "Target":       dict(color="#0072B2", ls="-",                   lw=1.6, z=7),
}
WINDOW_FACE = "#0072B2"
GUIDE_COLOR = "#1A1A1A"
GUIDE_LS = (0, (1, 1.2))

RC = {
    "font.size": BASE_FS, "axes.labelsize": BASE_FS, "axes.titlesize": BASE_FS,
    "xtick.labelsize": BASE_FS - 1, "ytick.labelsize": BASE_FS - 1,
    "legend.fontsize": BASE_FS - 1,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.solid_capstyle": "butt",
    "legend.frameon": False, "legend.handlelength": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600,
}


def find_effective_windows(t, P, target_index, threshold):
    """Contiguous runs of switching times at which the target wins.

    A switching time qualifies when the target is the argmax of the averaged
    final probabilities and reaches `threshold`. Returns
    (lo, hi, i_first, i_last); lo/hi are midpoints to the neighbouring failing
    grid points, so edges are resolved only to the grid spacing.
    """
    ok = (np.argmax(P, axis=1) == target_index) & (P[:, target_index] >= threshold)
    out, i, n = [], 0, len(t)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        lo = t[i] if i == 0 else 0.5 * (t[i - 1] + t[i])
        hi = t[j] if j == n - 1 else 0.5 * (t[j] + t[j + 1])
        out.append((float(lo), float(hi), i, j))
        i = j + 1
    return out


def make_figure(traj_probs, switch_step, sweep_t, sweep_P, roles,
                option_labels=None, x_max=None, x_max_b=None, threshold=0.98,
                fig_width=SINGLE_COL_W, panel_height=31 * MM,
                window_label="window of effective switching",
                name_roles_in_legend=False, savepath=None):
    """Two-row single-column figure. `roles` = [(role, signal_id), ...] in the
    order Source, Intermediate, Target. Returns (fig, (ax_a, ax_b), windows)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    target_index = roles[2][1]
    x_max = float(len(traj_probs) - 1) if x_max is None else float(x_max)
    x_max_b = x_max if x_max_b is None else float(x_max_b)
    windows = find_effective_windows(sweep_t, sweep_P, target_index, threshold)

    t_a = np.arange(len(traj_probs))
    keep_a = t_a <= x_max
    keep_b = sweep_t <= x_max_b

    with mpl.rc_context(RC):
        # Absolute margins: the canvas is exactly `fig_width` wide. Do not
        # re-save with bbox_inches="tight".
        m_left, m_right = 12 * MM, 4.0 * MM
        m_top, m_gap, m_bottom = 8.0 * MM, 13.5 * MM, 15.5 * MM
        fig_h = m_top + 2 * panel_height + m_gap + m_bottom
        fig = plt.figure(figsize=(fig_width, fig_h))
        ax_w = (fig_width - m_left - m_right) / fig_width
        ax_h = panel_height / fig_h
        x0 = m_left / fig_width
        ax_a = fig.add_axes([x0, 1 - (m_top + panel_height) / fig_h, ax_w, ax_h])
        ax_b = fig.add_axes([x0, m_bottom / fig_h, ax_w, ax_h])

        # -- a: trajectory at tau ----------------------------------------- #
        for role, idx in roles:
            s = ROLE_STYLE[role]
            ax_a.plot(t_a[keep_a], traj_probs[keep_a, idx], color=s["color"], clip_on=False,
                      ls=s["ls"], lw=s["lw"], zorder=s["z"])
        ax_a.axvline(switch_step, color=GUIDE_COLOR, ls=GUIDE_LS, lw=0.7, zorder=3)
        ax_a.annotate("", xy=(0, 1.08), xytext=(switch_step, 1.08),
                      xycoords=("data", "axes fraction"),
                      textcoords=("data", "axes fraction"),
                      arrowprops=dict(arrowstyle="<->", lw=0.6, color="#3C3C3C",
                                      shrinkA=0, shrinkB=0),
                      annotation_clip=False)
        ax_a.text(switch_step / 2, 1.11, r"$\tau$", ha="center", va="bottom",
                  transform=ax_a.get_xaxis_transform(), color="#3C3C3C")
        ax_a.set_xlabel("Time", labelpad=1.5)
        ax_a.set_ylabel("Adoption", labelpad=2)

        # -- b: sweep ------------------------------------------------------ #
        for lo, hi, *_ in windows:
            if lo > x_max_b:
                continue
            ax_b.axvspan(lo, min(hi, x_max_b), color=WINDOW_FACE, alpha=0.13,
                         lw=0, zorder=0)
        for role, idx in roles:
            s = ROLE_STYLE[role]
            ax_b.plot(sweep_t[keep_b], sweep_P[keep_b, idx], color=s["color"],
                      ls=s["ls"], lw=s["lw"], zorder=s["z"], clip_on=False,
                      drawstyle="steps-mid")
        ax_b.axvline(switch_step, color=GUIDE_COLOR, ls=GUIDE_LS, lw=0.7, zorder=3)
        ax_b.set_xlabel(r"Switching time ($\tau$)", labelpad=1.5)
        ax_b.set_ylabel("Final adoption", labelpad=2)

        vis = [w for w in windows if w[0] <= x_max_b]
        if vis and window_label:
            lo, hi = vis[int(np.argmax([min(w[1], x_max_b) - w[0] for w in vis]))][:2]
            hi_c = min(hi, x_max_b)
            ax_b.annotate("", xy=(lo, 1.08), xytext=(hi_c, 1.08),
                          xycoords=("data", "axes fraction"),
                          textcoords=("data", "axes fraction"),
                          arrowprops=dict(arrowstyle="<->" if hi <= x_max_b else "<-",
                                          lw=0.45, color=WINDOW_FACE,
                                          shrinkA=0, shrinkB=0, mutation_scale=4),
                          annotation_clip=False)
            if hi_c <= 0.55 * x_max_b:        # room on the right of the bracket
                ax_b.text(hi_c + 0.02 * x_max_b, 1.08, window_label, ha="left",
                          va="center", color=WINDOW_FACE, clip_on=False,
                          transform=ax_b.get_xaxis_transform(), fontsize=BASE_FS)
            else:                             # centred above the bracket
                ax_b.text(0.5 * (lo + hi_c), 1.11, window_label, ha="center",
                          va="bottom", color=WINDOW_FACE, clip_on=False,
                          transform=ax_b.get_xaxis_transform(), fontsize=BASE_FS)

        for ax, xm in ((ax_a, x_max), (ax_b, x_max_b)):
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 200)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticks(np.arange(0, 1.01, 0.25), minor=True)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax.spines["bottom"].set_position(("outward", 5))
            ax.spines["left"].set_position(("outward", 5))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(pad=1.5)

        letter_x = -m_left / (fig_width - m_left - m_right)
        for ax, letter in ((ax_a, "a"), (ax_b, "b")):
            ax.text(letter_x, 1.28, letter, transform=ax.transAxes,
                    fontsize=PANEL_FS, fontweight="bold", ha="left", va="top")

        def label(role, idx):
            if name_roles_in_legend and option_labels is not None:
                return f"{role} ({option_labels[idx]})"
            return role

        rh = [Line2D([], [], color=ROLE_STYLE[r]["color"], ls=ROLE_STYLE[r]["ls"],
                     lw=ROLE_STYLE[r]["lw"], label=label(r, i)) for r, i in roles]
        sw = Line2D([], [], color=GUIDE_COLOR, ls=GUIDE_LS, lw=0.7,
                    label="Switch point")
        blank = Line2D([], [], ls="none", label="")
        win = (Patch(facecolor=WINDOW_FACE, alpha=0.13, edgecolor="none",
                     label="Effective window") if vis else blank)
        # column-major fill: (Source, Intermediate) (Target, Switch) (Window, -)
        fig.legend(handles=[rh[0], rh[1], rh[2], sw, win, blank],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
                   borderaxespad=0.0, handletextpad=0.5, columnspacing=1.6,
                   labelspacing=0.35)

        if savepath:
            os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
            fig.savefig(savepath)
            if savepath.lower().endswith(".pdf"):
                fig.savefig(savepath[:-4] + ".png", dpi=600)
    return fig, (ax_a, ax_b), windows


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _load(path, fmt):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if d.get("format") != fmt:
        raise ValueError(f"unrecognised format in {path}")
    return d


def load_trajectory(path):
    return _load(path, TRAJ_FORMAT)


def load_sweep(path):
    return _load(path, SWEEP_FORMAT)


def _save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class Setup:
    """Everything derived from the shared CLI arguments."""

    def __init__(self, a, parser):
        self.names = [a.source, a.intermediate, a.target]
        if len(set(self.names)) != 3:
            parser.error("source, intermediate and target must be distinct")
        self.q_file, file_order = resolve_file(
            self.names, Q_PATTERN.format(shorthand=a.shorthand))
        if self.q_file is None:
            parser.error(f"no Q-dict found for any ordering of {self.names}")
        self.order = a.signal_order.split(",") if a.signal_order else file_order
        if sorted(self.order) != sorted(self.names):
            parser.error(f"--signal-order {self.order} is not a permutation "
                         f"of {self.names}")
        self.s_idx, self.b_idx, self.a_idx = (self.order.index(n) for n in self.names)
        self.dyn = ws.DynamicsParams(
            TIME=a.TIME, dt=a.dt, target_time_mode=a.target_time_mode,
            target_steps=a.target_steps, stall_detection=not a.no_stall)
        self.sig = ws.sig_hash(self.dyn)
        self.t_inter = None
        if getattr(a, "t_inter", None) is not None:
            self.t_inter = int(a.t_inter)
        elif getattr(a, "tau", None) is not None:
            self.t_inter = int(round(a.tau * a.TIME))
        tag = (f"{a.shorthand}_{a.source}_{a.intermediate}_{a.target}_"
               f"{a.H}mem_c{a.c:.4f}")
        self.traj_path = (None if self.t_inter is None else os.path.join(
            a.out_dir, f"TRAJ_{tag}_t{self.t_inter}_{self.sig}.pkl"))
        self.sweep_path = os.path.join(a.out_dir, f"SWEEP_{tag}_{self.sig}.pkl")
        self.roles = [("Source", self.s_idx), ("Intermediate", self.b_idx),
                      ("Target", self.a_idx)]

    def meta(self, a, **extra):
        direct_cm = None
        cm_file, _ = resolve_file(self.names,
                                  CM_PATTERN.format(shorthand=a.shorthand, H=a.H))
        if cm_file is not None:
            with open(cm_file, "rb") as f:
                cm = ws.get_critical_mass_data(pickle.load(f))
            direct_cm = ((cm.get(self.s_idx, {}) or {}).get(self.a_idx)
                         or {}).get("critical_mass")
        m = {
            "shorthand": a.shorthand, "H": a.H,
            "names": dict(zip(("source", "intermediate", "target"), self.names)),
            "signal_order": self.order,
            "signal_order_source": "argument" if a.signal_order else "q_file_name",
            "q_file": self.q_file,
            "indices": {"source": self.s_idx, "intermediate": self.b_idx,
                        "target": self.a_idx},
            "c": a.c, "direct_critical_mass": direct_cm,
            "dynamics": self.dyn.signature(), "signature": self.sig,
            "seed": getattr(a, "seed", None),
        }
        m.update(extra)
        return m


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(a, parser):
    S = Setup(a, parser)
    if a.seed is not None:
        np.random.seed(a.seed)
        random.seed(a.seed)
    with open(S.q_file, "rb") as f:
        ctx = ws.ModelContext(pickle.load(f), a.H)
    print(f"Q-dict: {S.q_file}\nsignal ids: "
          + ", ".join(f"{n}={i}" for i, n in enumerate(S.order))
          + f"\nS={S.s_idx} B={S.b_idx} A={S.a_idx}  c={a.c}  signature={S.sig}")

    # -- (i) trajectory at tau ------------------------------------------- #
    if S.t_inter is not None:
        if os.path.exists(S.traj_path) and not a.force:
            print(f"trajectory exists, skipping: {S.traj_path}")
        else:
            t0 = time.time()
            res = simulate(ctx, S.s_idx, S.b_idx, S.a_idx, a.c, S.t_inter,
                           S.dyn, a.stop_on_success, progress=not a.quiet)
            control = (simulate(ctx, S.s_idx, S.b_idx, S.a_idx, a.c, 0, S.dyn,
                                a.stop_on_success, progress=not a.quiet)
                       if a.control else None)
            cache = None
            if a.check_cache:
                w_file, _ = resolve_file(
                    S.names, WINDOW_PATTERN.format(shorthand=a.shorthand, H=a.H))
                cache = {"trajectory": check_against_cache(
                    res, w_file, S.dyn, S.s_idx, S.a_idx, a.c)}
                if control is not None:
                    cache["control"] = check_against_cache(
                        control, w_file, S.dyn, S.s_idx, S.a_idx, a.c)
            _save({"format": TRAJ_FORMAT,
                   "meta": S.meta(a, stop_on_success=a.stop_on_success,
                                  wall_seconds=round(time.time() - t0, 1)),
                   "trajectory": res, "control": control, "cache_check": cache},
                  S.traj_path)
            for lab, r in (("trajectory", res), ("control", control)):
                if r is not None:
                    print(f"{lab:>10}: t_inter={r['t_inter']} success={r['success']} "
                          f"t_success={r['t_success']} terminal={r['terminal']} "
                          f"final p={np.round(r['final_probs'], 4).tolist()} "
                          f"handover gap={r['handover_max_abs_logp_gap']}")
            for k, v in (cache or {}).items():
                print(f"cache[{k}]: {v['status']}")
            print(f"saved {S.traj_path}")

    # -- (ii) switching-time sweep --------------------------------------- #
    if a.no_sweep:
        return
    t_max = a.TIME - 1 if a.t_max is None else int(a.t_max)
    t_grid = np.arange(0, t_max + 1, a.t_stride, dtype=int)
    sweep_meta = S.meta(a, avg_window=a.avg_window, t_stride=a.t_stride,
                        t_max=t_max, snapshot_every=a.snapshot_every)

    done = {}
    if os.path.exists(S.sweep_path) and not a.force:
        old = load_sweep(S.sweep_path)
        if old["sweep"]["avg_window"] != a.avg_window:
            parser.error("existing sweep used a different --avg-window; "
                         "pass --force to recompute")
        sw = old["sweep"]
        done = {int(t): (m, lp, None if ts < 0 else int(ts)) for t, m, lp, ts in
                zip(sw["t_inter"], sw["mean_final_probs"],
                    sw["final_log_probs"], sw["t_success"])}
        missing = sum(int(t) not in done for t in t_grid)
        print(f"resuming sweep: {len(done)} cached, {missing} to run")
        if missing == 0:
            return

    def save_cb(d, inter_lp):
        _save({"format": SWEEP_FORMAT, "meta": sweep_meta,
               "sweep": assemble_sweep(d, sorted(set(d) | set(t_grid.tolist())),
                                       inter_lp, S.a_idx, S.b_idx, S.dyn,
                                       a.avg_window)},
              S.sweep_path)

    t0 = time.time()
    done, inter_lp = run_sweep(
        S.q_file, a.H, ctx, S.s_idx, S.b_idx, S.a_idx, a.c, S.dyn, t_grid,
        a.avg_window, every=a.snapshot_every, processes=a.processes,
        chunk=a.chunk, seed=a.seed, done=done, save_cb=save_cb,
        progress=not a.quiet)
    sweep_meta["wall_seconds"] = round(time.time() - t0, 1)
    sweep = assemble_sweep(done, t_grid, inter_lp, S.a_idx, S.b_idx, S.dyn,
                           a.avg_window)
    _save({"format": SWEEP_FORMAT, "meta": sweep_meta, "sweep": sweep},
          S.sweep_path)
    win = find_effective_windows(sweep["t_inter"], sweep["mean_final_probs"],
                                 S.a_idx, S.dyn.success_threshold)
    print(f"sweep: {len(sweep['t_inter'])} switching times, "
          f"{int(sweep['success_end'].sum())} successful; windows (steps): "
          f"{[(w[0], w[1]) for w in win]}\nsaved {S.sweep_path}")


def cmd_plot(a, parser):
    S = Setup(a, parser)
    traj_path = a.traj_file or S.traj_path
    sweep_path = a.sweep_file or S.sweep_path
    for pth, what in ((traj_path, "trajectory"), (sweep_path, "sweep")):
        if not os.path.exists(pth):
            parser.error(f"{what} file not found: {pth} (run the `run` "
                         f"command with the same arguments first)")
    traj = load_trajectory(traj_path)
    sw = load_sweep(sweep_path)["sweep"]
    tr = traj["trajectory"]

    probs = np.exp(tr["log_probs"])
    P = np.asarray(sw["mean_final_probs"], dtype=float)
    dev = max(np.abs(probs.sum(1) - 1).max(), np.abs(P.sum(1) - 1).max())
    print(f"max |sum(p) - 1| over plotted rows: {dev:.2e}"
          + ("  (renormalising)" if a.renormalise else ""))
    if a.renormalise:
        probs = probs / probs.sum(1, keepdims=True)
        P = P / P.sum(1, keepdims=True)
    if not sw.get("complete", True):
        print("warning: sweep is incomplete; plotting the switching times available")

    threshold = S.dyn.success_threshold if a.threshold is None else a.threshold
    savepath = a.savepath or os.path.join(
        "figures", f"temporal_dynamics_critical_mass_{a.shorthand}_{a.source}_"
                   f"{a.intermediate}_{a.target}_c{a.c:.4f}_t{tr['t_inter']}.pdf")
    _, _, windows = make_figure(
        probs, tr["t_inter"], np.asarray(sw["t_inter"], dtype=float), P, S.roles,
        option_labels=S.order, x_max=a.x_max, x_max_b=a.x_max_b,
        threshold=threshold, panel_height=a.panel_height * MM,
        window_label=None if a.no_window_label else "window of effective switching",
        name_roles_in_legend=a.name_roles, savepath=savepath)
    T = S.dyn.TIME
    for lo, hi, *_ in windows:
        print(f"window: t in [{lo:g}, {hi:g}] steps  (tau in [{lo / T:.4f}, {hi / T:.4f}])")
    if not windows:
        print(f"no effective window at threshold {threshold}")
    print(f"saved {savepath}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--shorthand", default="dsR1_Q32B")
    common.add_argument("--H", type=int, default=3)
    common.add_argument("--source", required=True, help="incumbent convention S")
    common.add_argument("--intermediate", required=True, help="stepping stone B")
    common.add_argument("--target", required=True, help="target convention A")
    common.add_argument("--c", type=float, required=True, help="committed fraction")
    g = common.add_mutually_exclusive_group()
    g.add_argument("--tau", type=float, help="switching time / TIME (panel a)")
    g.add_argument("--t-inter", type=int, help="switching time in steps (panel a)")
    common.add_argument("--signal-order", default=None,
                        help="comma-separated names in signal-id order")
    # dynamics: defaults mirror `window_search_v2.py run`
    common.add_argument("--time", type=int, default=1000, dest="TIME")
    common.add_argument("--dt", type=float, default=0.1)
    common.add_argument("--target-time-mode", default="fixed_total",
                        choices=["fixed_budget", "fixed_total"])
    common.add_argument("--target-steps", type=int, default=None)
    common.add_argument("--no-stall", action="store_true",
                        help="signature only; stall detection is never applied")
    common.add_argument("--out-dir", default="meta_data_async/mean_field_trajectories")

    p = argparse.ArgumentParser(description="Stepping-stone trajectory and "
                                            "switching-time sweep for one triple.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common], help="simulate and record")
    r.add_argument("--no-sweep", action="store_true",
                   help="only the trajectory at --tau")
    r.add_argument("--t-stride", type=int, default=1,
                   help="switching-time spacing in steps (default: every step)")
    r.add_argument("--t-max", type=int, default=None,
                   help="largest switching time in steps (default: TIME - 1)")
    r.add_argument("--avg-window", type=int, default=3,
                   help="final steps over which probabilities are averaged")
    r.add_argument("--snapshot-every", type=int, default=1,
                   help="store the intermediate population every k steps")
    r.add_argument("--processes", type=int, default=1)
    r.add_argument("--chunk", type=int, default=10,
                   help="switching times per worker job")
    r.add_argument("--stop-on-success", action="store_true",
                   help="trajectory only; the sweep always runs the full budget")
    r.add_argument("--control", action="store_true",
                   help="also record the direct trajectory (t_inter = 0)")
    r.add_argument("--check-cache", action="store_true")
    r.add_argument("--force", action="store_true", help="recompute existing files")
    r.add_argument("--seed", type=int, default=None,
                   help="seeds numpy/random; relevant only for stochastic dynamics")
    r.add_argument("--quiet", action="store_true")

    q = sub.add_parser("plot", parents=[common], help="make the figure")
    q.add_argument("--traj-file", default=None)
    q.add_argument("--sweep-file", default=None)
    q.add_argument("--x-max", type=float, default=None,
                   help="x-limit of panel a, in steps (default: full trajectory)")
    q.add_argument("--x-max-b", type=float, default=None,
                   help="x-limit of panel b (default: --x-max)")
    q.add_argument("--threshold", type=float, default=None,
                   help="target adoption defining the window "
                        "(default: success_threshold)")
    q.add_argument("--panel-height", type=float, default=31.0, help="mm")
    q.add_argument("--name-roles", action="store_true",
                   help="legend reads 'Source (sadness)' etc.")
    q.add_argument("--no-window-label", action="store_true")
    q.add_argument("--renormalise", action="store_true",
                   help="divide probabilities by their row sum before plotting")
    q.add_argument("--savepath", default=None)

    a = p.parse_args(argv)
    if a.cmd == "plot" and a.tau is None and a.t_inter is None:
        q.error("plot requires --tau or --t-inter")
    if a.cmd == "run" and a.snapshot_every < 1:
        r.error("--snapshot-every must be >= 1")
    (cmd_run if a.cmd == "run" else cmd_plot)(a, p)


if __name__ == "__main__":
    main()

# %%