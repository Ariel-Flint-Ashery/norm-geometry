#%%
import argparse
import copy
import csv
import multiprocessing as mp
import os
import pickle
import random
from itertools import permutations
from tqdm import tqdm
import numpy as np

import mean_field_module as mfm

WINDOW = 3
LABELS = ("source", "intermediate", "target")
FIELDS = ["model", "H", "source", "intermediate", "target",
          "source_name", "intermediate_name", "target_name",
          "status", "critical_mass", "T_c", "n_tau",
          "entrenched_intermediate", "entrenched_target", "full_commit_label"]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def find_matrix_file(names, pattern):
    """Return (path, names in the order used by the file name).

    The names are returned in file-name order because a permuted file may be the
    one on disk. ASSUMPTION: signal index i in the file corresponds to the i-th
    name in its file name. Verify this against mean_field_module.H_reducer.
    """
    for perm in permutations(names, 3):
        path = pattern.format(name1=perm[0], name2=perm[1], name3=perm[2])
        if os.path.exists(path):
            return path, tuple(perm)
    raise FileNotFoundError(f"No file matching {pattern} for {names}")


def get_critical_mass(cm_data, s, a):
    """Return (status, critical_mass, T_c). Raises if no solution was recorded."""
    try:
        sol = cm_data[s][a]["solution"]
    except KeyError as e:
        raise KeyError(f"No critical-mass solution for source {s} -> target {a}") from e
    if sol["critical_mass"] is None:
        return "no_critical_mass", None, None
    if sol["time"] is None:
        return "no_critical_time", sol["critical_mass"], None
    return "ok", float(sol["critical_mass"]), int(sol["time"])


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #
class Model:
    """Reduced-state quantities shared by every simulation of one triplet."""

    def __init__(self, q_total, H):
        q_H, keys_H, s0, s1, s2 = mfm.H_reducer(q_total, H)
        F, Finv = mfm.integer_mapping(keys_H)
        self.choices, self.nn_choices = mfm.reverse_shift_vectorized(Finv)
        self.q_s = mfm.integer_probabilities(q_H, F)
        self.empty_state = F[""]
        self.Ss = [F[s0], F[s1], F[s2]]
        self.P_key, self.P_value = mfm.state_transitions(q_H, H, F, Finv)

    def llm(self, s, signal, c, population):
        return mfm.LLM_dynamics_integer(
            q_s=self.q_s, P_key=self.P_key, P_value=self.P_value,
            empty_state=self.empty_state, choices=self.choices,
            nn_choices=self.nn_choices, steady_state=self.Ss[s],
            CM=c, committment_signal=signal, population=population,
            dt = 0.1
        )


def _copy(p):
    return p.copy() if isinstance(p, np.ndarray) else copy.deepcopy(p)


def committed_snapshots(model, s, a, c, T_c):
    """Populations after 0, 1, ..., T_c committed updates (length T_c + 1)."""
    llm = model.llm(s, a, c, None)
    llm.initialize_population_steady()
    snaps = [_copy(llm.population)]
    for _ in range(T_c):
        llm.algorithmic_update_asynchronous()
        snaps.append(_copy(llm.population))
    return snaps


def relax(model, s, a, population, n_steps):
    """Evolve without commitment; return the last WINDOW log-prob vectors."""
    llm = model.llm(s, a, 0.0, _copy(population))
    hist = np.full((WINDOW, 3), -np.inf)
    for t in range(n_steps):
        llm.algorithmic_update_asynchronous()
        hist[t % WINDOW] = llm.population_log_probs
    return hist


def label_end_state(hist, s, b, a, threshold):
    """Mean over the window (axis 0) of probabilities, not of log-probs."""
    mean_p = np.exp(hist).mean(axis=0)
    for name, idx in zip(LABELS, (s, b, a)):
        if mean_p[idx] >= threshold:
            return name
    return "mixed"


def run_pair(model, s, a, c, T_c, taus, total_steps, threshold):
    """Commit for k steps, then relax for total_steps - k (fixed horizon)."""
    b = 3 - s - a
    snaps = committed_snapshots(model, s, a, c, T_c)
    # steps = sorted({int(t * T_c) for t in taus if t < 1.0})
    steps = list(range(1, T_c))  # all steps except 0 and T_c
    labels = {k: label_end_state(relax(model, s, a, snaps[k], T_c - k),
                                 s, b, a, threshold) for k in tqdm(steps, desc="Building labels")}
    # useless, but keep anyway
    full = label_end_state(relax(model, s, a, snaps[T_c], 0),
                           s, b, a, threshold)
    return {
        "n_tau": len(steps),
        "entrenched_intermediate": any(v == "intermediate" for v in labels.values()),
        "entrenched_target": any(v == "target" for v in labels.values()),
        "full_commit_label": full,
    }


# --------------------------------------------------------------------------- #
# one triplet (one worker)
# --------------------------------------------------------------------------- #
def run_triplet(job):
    options, opt_id, args = job
    seed = os.getpid() + opt_id
    random.seed(seed)
    np.random.seed(seed)

    q_file, q_names = find_matrix_file(
        options, f"policies/log_q_dicts/Q_dict_{args.model}"
        + "_{name1}_{name2}_{name3}_0.5tmp.pkl")
    cm_file, cm_names = find_matrix_file(
        options, f"meta_data_async/mean_field_critical_mass/LLM_dynamics_{args.model}"
        + "_{name1}_{name2}_{name3}_" + f"{args.H}mem_0.5tmp.pkl")
    if q_names != cm_names:
        raise ValueError(f"Signal order differs between {q_file} and {cm_file}")

    cache = os.path.join(args.out_dir, "rows",
                         f"{args.model}_{'_'.join(q_names)}_{args.H}mem"
                         f"_T{args.total_steps}_r{args.tau_resolution}"
                         f"_th{args.threshold}.pkl")
    if os.path.exists(cache) and not args.overwrite:
        with open(cache, "rb") as f:
            return pickle.load(f)

    with open(q_file, "rb") as f:
        model = Model(pickle.load(f), args.H)
    with open(cm_file, "rb") as f:
        cm_data = pickle.load(f)

    n = int(round(1 / args.tau_resolution))
    taus = [i / n for i in range(n + 1)]

    rows = []
    for s in range(3):
        for a in range(3):
            if a == s:
                continue
            b = 3 - s - a
            status, c, T_c = get_critical_mass(cm_data, s, a)
            if status == "ok" and args.total_steps - T_c < WINDOW:
                status = "total_too_short"
            row = dict.fromkeys(FIELDS, "")
            row.update(model=args.model, H=args.H, source=s, intermediate=b,
                       target=a, source_name=q_names[s],
                       intermediate_name=q_names[b], target_name=q_names[a],
                       status=status, critical_mass=c, T_c=T_c)
            if status == "ok":
                row.update(run_pair(model, s, a, c, T_c, taus,
                                    args.total_steps, args.threshold))
            rows.append(row)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = f"{cache}.tmp_{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(rows, f)
    os.replace(tmp, cache)
    print(f"[{os.getpid()}] done {q_names}")
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama32_3B")
    p.add_argument("--H", type=int, default=3)
    p.add_argument("--triplets", default="policies/emotion_triplets.pkl")
    p.add_argument("--out-dir", default="meta_data_async/mean_field_experiment_1")
    p.add_argument("--tau-resolution", type=float, default=0.01)
    p.add_argument("--total-steps", type=int, default=1000,
                   help="fixed observation horizon: commitment + relaxation")
    p.add_argument("--threshold", type=float, default=0.98)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--limit", type=int, default=None, help="first N triplets only")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    with open(args.triplets, "rb") as f:
        triplets = pickle.load(f)[: args.limit]
    jobs = [(opts, i, args) for i, opts in enumerate(triplets)]
    n_proc = max(1, min(args.processes, len(jobs)))
    with mp.Pool(n_proc) as pool:
        results = pool.map(run_triplet, jobs)

    out = os.path.join(args.out_dir, f"entrenchment_{args.model}_{args.H}mem"
                                     f"_T{args.total_steps}.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in results:
            w.writerows(rows)
    print(f"Wrote {sum(map(len, results))} rows to {out}")


if __name__ == "__main__":
    main()