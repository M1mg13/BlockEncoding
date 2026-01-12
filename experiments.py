#!/usr/bin/env python3
# Copyright 2025 Fraunhofer
# Licensed under the Apache License, Version 2.0 (the "License");
"""
experiments.py

Experiment driver for BlockEncoding project.

- Produces per-instance JSON summaries (results_full/).
- Produces Figures (Images/).
- SVD guard enforces conservative memory usage (default ~30 GiB).
- Mode: 'results' (only compute JSONs), 'plots' (only plotting), 'all' (both).

Run examples:
  python3 experiments.py                          # default: all
  python3 experiments.py --mode results --repeat 3
  python3 experiments.py --mode plots
"""
import argparse
import json
import math
import random
import traceback
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import BlockEncoding as BE

# Aliases from library

TensorNetworkEncoding = BE.TensorNetworkEncoding
QUBOSweepEncoding = BE.QUBOSweepEncoding
QUBOPauliSumEncoding = BE.QUBOPauliSumEncoding
QUBONetworkEncoding = BE.QUBONetworkEncoding
SiteLayer = BE.SiteLayer
MonkeyPatch = BE.MonkeyPatch
base_count = BE.base_count

# Output locations

OUTDIR = Path("results_full")
IMGDIR = Path("Images")
OUTDIR.mkdir(exist_ok=True)
IMGDIR.mkdir(exist_ok=True)

# Defaults

REPEAT = 5
SEED = 1234
BYTES_PER_ELEMENT = np.dtype(np.complex128).itemsize
DEFAULT_ALLOWED_GB = 30.0
DEFAULT_MEM_FACTOR = 6.0  # safety factor for extra workspace beyond singular vectors

# SVD guard

_orig_svd = np.linalg.svd


def _svd_guard(a, full_matrices=True, compute_uv=True, **kw):
    shape = getattr(a, "shape", None)
    if shape and len(shape) == 2:
        m, n = int(shape[0]), int(shape[1])
        metric = m * m + n * n
        max_elems = getattr(_svd_guard, "max_elements", None)
        if max_elems is None:
            raise RuntimeError("SVD guard not configured (set _svd_guard.max_elements)")
        if metric > max_elems:
            raise MemoryError(
                f"SVD metric m^2+n^2={metric} exceeds threshold {max_elems}"
            )
    return _orig_svd(a, full_matrices=full_matrices, compute_uv=compute_uv, **kw)


# Instance generators


def gen_peps_grid(r, chi=2, seed=0, pad_pow=2):
    n = r * r
    adj = {i: [] for i in range(n)}
    for i in range(n):
        row, col = divmod(i, r)
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            rr, cc = row + dr, col + dc
            if 0 <= rr < r and 0 <= cc < r:
                adj[i].append(rr * r + cc)
    return _make_graph_tne(adj, bond_dim=chi, seed=seed, pad_pow=pad_pow)


def gen_random_graph(n, avg_deg=3, chi=2, seed=0, pad_pow=2):
    p = float(avg_deg) / max(1, n - 1)
    rng = np.random.default_rng(seed)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].append(j)
                adj[j].append(i)
    return _make_graph_tne(adj, bond_dim=chi, seed=seed, pad_pow=pad_pow)


def _make_graph_tne(adj, bond_dim=4, phys=2, seed=0, pad_pow=2):
    rng = np.random.default_rng(seed)
    neighs = {i: sorted(adj[i]) for i in range(len(adj))}
    bond_dims = {}
    for i in range(len(adj)):
        for j in neighs[i]:
            if i < j:
                bond_dims[(i, j)] = int(bond_dim)
    V = []
    for i in range(len(adj)):
        axes = [bond_dims[(min(i, j), max(i, j))] for j in neighs[i]]
        shape = tuple(axes) + (phys, phys)
        T = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        V.append(T)
    E = -np.ones((len(adj), len(adj)), dtype=int)
    for i in range(len(adj)):
        for ai, j in enumerate(neighs[i]):
            E[i, j] = ai
    return TensorNetworkEncoding(V, E, pad_pow=pad_pow)


def gen_qubo_sweep(n, seed=0):
    rng = np.random.default_rng(seed)
    Q = 0.5 * (rng.normal(size=(n, n)) + rng.normal(size=(n, n)).T)
    return QUBOSweepEncoding(Q, c=0.0, tol=1e-12)


# Processing function that captures SiteFactor returns and returns a record


def process_instance(tne, family, params, max_dense_dim=(1 << 14)):
    captured = []
    id_to_index = {id(v): i for i, v in enumerate(tne.V)}

    # wrapper around SiteLayer.factor to capture SiteFactor
    orig_factor = SiteLayer.factor

    def wrapper(self, *args, **kwargs):
        sf = orig_factor(self, *args, **kwargs)
        idx = id_to_index.get(id(self), None)
        captured.append((idx, sf))
        return sf

    with MonkeyPatch(SiteLayer, "factor", wrapper):
        res = tne.build_operator_sequence(unfold=True, tol=1e-12, diluted_cores=True)

    if len(res) == 4:
        Ops, regs_atoms, gamma, _ = res
    else:
        Ops, regs_atoms, gamma = res

    # per-site metadata
    n_sites = len(tne.V)
    per_site = [None] * n_sites
    for idx, sf in captured:
        per_site[idx] = {
            "m": int(sf.m),
            "n": int(sf.n),
            "beta": float(sf.beta),
            "svals": (
                sf.svals.tolist() if hasattr(sf.svals, "tolist") else list(sf.svals)
            ),
        }

    # fallbacks if not all sites captured
    for i in range(n_sites):
        if per_site[i] is None:
            per_site[i] = {"m": None, "n": None, "beta": None, "svals": None}

    gamma_log = (
        float(math.log(max(1e-300, float(gamma)))) if gamma is not None else None
    )

    # peak unfolding
    max_unfold = 0
    for p in per_site:
        try:
            if p.get("m"):
                max_unfold = max(max_unfold, int(p["m"]))
            if p.get("n"):
                max_unfold = max(max_unfold, int(p["n"]))
        except Exception:
            pass
    if max_unfold == 0 and Ops:
        for t, targ in Ops:
            if hasattr(t, "shape"):
                max_unfold = max(max_unfold, int(max(t.shape)))

    pad_pow = int(tne.pad_pow)
    num_qudits = int(regs_atoms) if regs_atoms is not None else None
    physical_atoms = sum(base_count(int(v.T.shape[-1]), pad_pow) for v in tne.V)
    ancilla_atoms = int(num_qudits - physical_atoms) if num_qudits is not None else None
    if ancilla_atoms is not None and ancilla_atoms < 0:
        ancilla_atoms = 0
    ancilla_qubits = (
        int(ancilla_atoms)
        if pad_pow == 2 and ancilla_atoms is not None
        else (
            float(ancilla_atoms) * math.log2(pad_pow)
            if ancilla_atoms is not None
            else None
        )
    )

    betas = [p["beta"] for p in per_site if p and p.get("beta") is not None]
    beta_mean = float(np.mean(betas)) if betas else None
    beta_min = float(np.min(betas)) if betas else None

    # optional dense checks when feasible (and cheap)
    beta_H = None
    p_haar_mean = None
    try:
        total_dim = pad_pow ** int(num_qudits) if num_qudits is not None else 1 << 30
        if total_dim <= max_dense_dim:
            H, _ = tne.build_dense_matrix(max_dims=max_dense_dim, tol=1e-12)
            svals = np.linalg.svd(H, compute_uv=False)
            beta_H = float(svals.max())
            dim = H.shape[0]
            rng = np.random.default_rng(0)
            psis = [rng.normal(size=dim) + 1j * rng.normal(size=dim) for _ in range(32)]
            psis = [v / np.linalg.norm(v) for v in psis]
            A = H / float(gamma) if gamma is not None else H
            probs = [float(np.linalg.norm(A @ psi) ** 2) for psi in psis]
            p_haar_mean = float(np.mean(probs))
    except MemoryError:
        pass

    rec = {
        "family": family,
        "params": params,
        "regs_atoms": int(regs_atoms) if regs_atoms is not None else None,
        "pad_pow": pad_pow,
        "num_qudits": num_qudits,
        "physical_atoms": int(physical_atoms),
        "ancilla_atoms": int(ancilla_atoms) if ancilla_atoms is not None else None,
        "ancilla_qubits": ancilla_qubits,
        "gamma_log": gamma_log,
        "beta_mean": beta_mean,
        "beta_min": beta_min,
        "beta_H": beta_H,
        "p_haar_mean": p_haar_mean,
        "site_count": n_sites,
        "max_unfold_dim": int(max_unfold),
        "per_site": per_site,
    }
    return rec


# Runner helper: writes JSON and returns record or skip/fail marker


def run_one_and_write(func, outpath: Path, family: str, params: dict):
    try:
        rec = func()
        rec.setdefault("family", family)
        rec.setdefault("params", params)
        outpath.write_text(json.dumps(rec, indent=2))
        print(f"WROTE {outpath.name} OK")
        return "ok", rec
    except MemoryError as me:
        msg = str(me)
        skip_rec = {
            "family": family,
            "params": params,
            "skipped_heavy": True,
            "skip_reason": msg,
        }
        outpath.write_text(json.dumps(skip_rec, indent=2))
        print(f"WROTE {outpath.name} SKIPPED: {msg}")
        return "skipped", skip_rec
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        fail_rec = {
            "family": family,
            "params": params,
            "failed": True,
            "skip_reason": str(e),
            "traceback": tb,
        }
        outpath.write_text(json.dumps(fail_rec, indent=2))
        print(f"WROTE {outpath.name} FAILED: {e}")
        return "failed", fail_rec


# Small helper to attempt multiple QUBO encoders for a given n (stops on SVD guard)


def try_qubo_encoders(
    n, seed, outdir, allowed_encoders=("sweep", "pauli", "network"), repeat=1
):
    results = []
    enc_map = {
        "sweep": QUBOSweepEncoding,
        "pauli": QUBOPauliSumEncoding,
        "network": QUBONetworkEncoding,
    }
    for enc in allowed_encoders:
        if enc not in enc_map:
            continue
        for inst in range(repeat):
            params = {"n": n, "inst": inst, "encoder": enc}
            outpath = outdir / f"qubo_{enc}_n{n}_inst{inst}.json"

            def build():
                if enc == "sweep":
                    return process_instance(
                        gen_qubo_sweep(n, seed + inst), f"QUBO_{enc}", params
                    )
                else:
                    rng = np.random.default_rng(seed + inst)
                    Q = 0.5 * (rng.normal(size=(n, n)) + rng.normal(size=(n, n)).T)
                    if enc == "pauli":
                        tne = QUBOPauliSumEncoding(Q, c=0.0, tol=1e-12)
                    else:
                        tne = QUBONetworkEncoding(Q, c=0.0, tol=1e-12)
                    return process_instance(tne, f"QUBO_{enc}", params)

            status, rec = run_one_and_write(build, outpath, f"QUBO_{enc}", params)
            results.append(rec)
    return results


# Sweep-order ablation: patch tne.next_site to alternative heuristics and run


def sweep_order_ablation(
    tne, name_prefix, outdir, orders=("left", "greedy", "random"), seed=0
):
    results = []

    def make_next(order):
        if order == "left":
            return lambda remain: min(remain)
        if order == "greedy":
            return tne.next_site
        if order == "random":
            rng = random.Random(seed)
            return lambda remain: rng.choice(sorted(remain))
        raise ValueError("unknown order")

    for order in orders:
        params = {"order": order}
        outpath = outdir / f"{name_prefix}_order_{order}.json"
        orig_next = tne.next_site
        try:
            tne.next_site = make_next(order)
            rec = process_instance(tne, f"{name_prefix}_order", params)
            outpath.write_text(json.dumps(rec, indent=2))
            print(f"WROTE {outpath.name} OK (order={order})")
            results.append(rec)
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            err = {
                "failed": True,
                "params": params,
                "skip_reason": str(e),
                "traceback": tb,
            }
            outpath.write_text(json.dumps(err, indent=2))
            print(f"WROTE {outpath.name} FAILED (order={order}): {e}")
            results.append(err)
        finally:
            tne.next_site = orig_next
    return results


# Top-level sweep: run families, qubo encoders, and some ablations


def run_all(outdir: Path, repeat=REPEAT, seed=SEED):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    # PEPS: (r, chi) grid
    for r, chi in product([3, 4], [2, 4]):
        for inst in range(repeat):
            params = {"r": r, "chi": chi, "inst": inst}
            outpath = outdir / f"peps_r{r}_chi{chi}_inst{inst}.json"
            status, rec = run_one_and_write(
                lambda: process_instance(
                    gen_peps_grid(r, chi, seed=seed + inst), "PEPS_grid", params
                ),
                outpath,
                "PEPS_grid",
                params,
            )
            rows.append(rec)

    # Random graphs
    for n, d in product([12, 16, 20], [2, 3]):
        for inst in range(repeat):
            params = {"n": n, "avg_deg": d, "inst": inst}
            outpath = outdir / f"rg_n{n}_deg{d}_inst{inst}.json"
            status, rec = run_one_and_write(
                lambda: process_instance(
                    gen_random_graph(n, d, chi=4, seed=seed + inst),
                    "RandomGraph",
                    params,
                ),
                outpath,
                "RandomGraph",
                params,
            )
            rows.append(rec)

    # MPO QUBO sweep (left-to-right MPO encoding)
    for n in [12, 16, 20]:
        for inst in range(repeat):
            params = {"n": n, "inst": inst}
            outpath = outdir / f"qubo_sweep_n{n}_inst{inst}.json"
            status, rec = run_one_and_write(
                lambda: process_instance(
                    gen_qubo_sweep(n, seed + inst), "QUBO_sweep", params
                ),
                outpath,
                "QUBO_sweep",
                params,
            )
            rows.append(rec)

    # QUBO encoder comparisons (smaller n)
    for n in [8, 10, 12]:
        results = try_qubo_encoders(
            n,
            seed=seed,
            outdir=outdir,
            allowed_encoders=("sweep", "pauli", "network"),
            repeat=min(3, repeat),
        )
        rows.extend(results)

    # Sweep-order ablation on representative instances
    try:
        tne_peps = gen_peps_grid(4, chi=2, seed=seed)
        rows.extend(
            sweep_order_ablation(
                tne_peps,
                "PEPS_r4_chi2",
                outdir,
                orders=("left", "greedy", "random"),
                seed=seed,
            )
        )
    except Exception as e:
        print("PEPS ablation failed:", e)

    try:
        tne_qubo = gen_qubo_sweep(12, seed=seed)
        rows.extend(
            sweep_order_ablation(
                tne_qubo,
                "QUBO_sweep_n12",
                outdir,
                orders=("left", "greedy", "random"),
                seed=seed,
            )
        )
    except Exception as e:
        print("QUBO ablation failed:", e)

    # summary
    (outdir / "summary_full.json").write_text(json.dumps(rows, indent=2))

    # CSV summary
    csvpath = outdir / "summary_full.csv"
    df_rows = []
    for r in rows:
        row = {
            k: r.get(k)
            for k in [
                "family",
                "params",
                "regs_atoms",
                "physical_atoms",
                "ancilla_atoms",
                "ancilla_qubits",
                "gamma_log",
                "beta_mean",
                "beta_min",
                "max_unfold_dim",
                "site_count",
                "skipped_heavy",
                "failed",
            ]
        }
        row["params"] = json.dumps(r.get("params", {}))
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(csvpath, index=False)
    print("WROTE summary:", csvpath)
    return rows


# Compact plotting for the paper (improved visuals + warnings suppressed)


def make_plots(outdir: Path, images: Path):
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(outdir.glob("*.json")):
        if p.name.startswith("summary"):
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(j, dict):
            rows.append(j)
        elif isinstance(j, list):
            rows.extend([x for x in j if isinstance(x, dict)])
    if not rows:
        print("No JSON results found in", outdir)
        return

    df = pd.DataFrame(rows)

    # Robust boolean mask helper to avoid pandas FutureWarning and ensure booleans
    def _bool_mask(colname, default=False):
        if colname not in df.columns:
            return pd.Series(default, index=df.index)
        s = df[colname].copy()
        return s.apply(lambda v: bool(v) if pd.notnull(v) else default)

    mask_skipped = _bool_mask("skipped_heavy", default=False)
    mask_failed = _bool_mask("failed", default=False)
    df_ok = df.loc[~mask_skipped & ~mask_failed].copy()
    if df_ok.empty:
        print("No completed results to plot.")
        return

    sns.set(style="whitegrid", context="paper")

    def fig_width_for_categories(n_cat, base=6, per_cat=0.6, minw=6, maxw=18):
        return min(max(base + n_cat * per_cat, minw), maxw)

    # Fig 1: ancilla qubits vs peak unfolding (scatter)
    if {"max_unfold_dim", "ancilla_qubits"}.issubset(df_ok.columns):
        plt.figure(figsize=(7, 4.5))
        ax = sns.scatterplot(
            data=df_ok,
            x="max_unfold_dim",
            y="ancilla_qubits",
            hue="family",
            s=80,
            alpha=0.9,
            palette="tab10",
            edgecolor="w",
            linewidth=0.6,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Peak unfolding dimension (log)")
        ax.set_ylabel("Ancilla qubits (log)")
        if ax.get_legend():
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(
            images / "ancilla_vs_unfold_scatter_full.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Fig 2: ancilla per family (box + jittered points)
    if "ancilla_qubits" in df_ok.columns:
        fams = sorted(df_ok["family"].astype(str).unique())
        w = fig_width_for_categories(len(fams), base=4, per_cat=0.6)
        plt.figure(figsize=(w, 4))
        sns.set_palette("pastel")
        ax = sns.boxplot(data=df_ok, x="family", y="ancilla_qubits", showfliers=False)
        # overlay jittered points (no hue to avoid duplicated legends)
        sns.stripplot(
            data=df_ok,
            x="family",
            y="ancilla_qubits",
            color="k",
            size=4,
            jitter=0.18,
            alpha=0.8,
        )
        ax.set_ylabel("Ancilla qubits")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        if ax.get_legend():
            # ensure legend removed if present
            ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(images / "ancilla_by_family_full.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 3: gamma_log by family (box + points)
    if "gamma_log" in df_ok.columns:
        fams = sorted(df_ok["family"].astype(str).unique())
        w = fig_width_for_categories(len(fams), base=4, per_cat=0.6)
        plt.figure(figsize=(w, 3.6))
        sns.set_palette("pastel")
        ax = sns.boxplot(data=df_ok, x="family", y="gamma_log", showfliers=False)
        sns.stripplot(
            data=df_ok,
            x="family",
            y="gamma_log",
            color="k",
            size=3,
            jitter=0.18,
            alpha=0.6,
        )
        ax.set_ylabel("log(Gamma)")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        if ax.get_legend():
            ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(images / "gamma_log_by_family.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 4: QUBO encoder comparison
    if "family" in df_ok.columns:
        qubo_df = df_ok[df_ok["family"].astype(str).str.contains("QUBO")]
    else:
        qubo_df = pd.DataFrame()
    if not qubo_df.empty:
        qubo_df = qubo_df.copy()
        qubo_df["encoder"] = qubo_df["family"].str.replace("QUBO_", "", regex=False)
        encs = sorted(qubo_df["encoder"].unique())
        w = fig_width_for_categories(len(encs), base=4, per_cat=1.2)

        # ancilla comparison
        plt.figure(figsize=(w, 3.4))
        sns.set_palette("Set2")
        ax = sns.boxplot(
            data=qubo_df, x="encoder", y="ancilla_qubits", showfliers=False
        )
        sns.stripplot(
            data=qubo_df,
            x="encoder",
            y="ancilla_qubits",
            color="k",
            size=4,
            jitter=0.15,
            alpha=0.7,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Ancilla qubits (log)")
        ax.set_xlabel("QUBO encoder")
        if ax.get_legend():
            ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(images / "qubo_encoder_ancilla.png", dpi=300, bbox_inches="tight")
        plt.close()

        # unfolding comparison
        plt.figure(figsize=(w, 3.4))
        sns.set_palette("Set2")
        ax = sns.boxplot(
            data=qubo_df, x="encoder", y="max_unfold_dim", showfliers=False
        )
        sns.stripplot(
            data=qubo_df,
            x="encoder",
            y="max_unfold_dim",
            color="k",
            size=4,
            jitter=0.15,
            alpha=0.7,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Peak unfolding (log)")
        ax.set_xlabel("QUBO encoder")
        if ax.get_legend():
            ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(images / "qubo_encoder_unfold.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 5: sweep-order ablation (box + points)
    if "family" in df_ok.columns:
        ablation_df = df_ok[df_ok["family"].astype(str).str.contains("_order")]
    else:
        ablation_df = pd.DataFrame()
    if not ablation_df.empty:
        ab = ablation_df.copy()
        ab["base"] = ab["family"].astype(str).str.replace(r"_order.*", "", regex=True)

        def _get_order(val):
            try:
                pp = val
                if isinstance(pp, str):
                    pp = json.loads(pp)
                return pp.get("order")
            except Exception:
                if isinstance(val, dict):
                    return val.get("params", {}).get("order")
                return None

        if "params" in ab.columns:
            ab["order"] = ab["params"].apply(_get_order)
        else:
            ab["order"] = None

        ab = ab[ab["order"].notnull()].copy()
        if not ab.empty and "gamma_log" in ab.columns:
            order_cats = ["left", "greedy", "random"]
            ab["order"] = pd.Categorical(
                ab["order"], categories=order_cats, ordered=True
            )
            bases = sorted(ab["base"].unique())
            w = fig_width_for_categories(len(bases), base=5, per_cat=1.0)
            plt.figure(figsize=(w, 3.6))
            ax = sns.boxplot(
                data=ab,
                x="base",
                y="gamma_log",
                hue="order",
                palette="Set1",
                showfliers=False,
            )
            # overlay points, dodge by hue to match boxes
            sns.stripplot(
                data=ab,
                x="base",
                y="gamma_log",
                hue="order",
                dodge=True,
                palette="Set1",
                size=4,
                jitter=0.18,
                alpha=0.9,
                edgecolor="k",
                linewidth=0.2,
            )
            # de-duplicate and move legend outside
            if ax.get_legend():
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                unique = dict(zip(labels, handles))
                plt.legend(
                    unique.values(),
                    unique.keys(),
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
            ax.set_ylabel("log(Gamma)")
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
            plt.tight_layout()
            plt.savefig(
                images / "ablation_gamma_by_order.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    print("Saved plots to", images)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--images", type=str, default=str(IMGDIR))
    p.add_argument("--repeat", type=int, default=REPEAT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--allowed-mem-gb", type=float, default=DEFAULT_ALLOWED_GB)
    p.add_argument("--mem-factor", type=float, default=DEFAULT_MEM_FACTOR)
    p.add_argument(
        "--mode",
        choices=("all", "results", "plots"),
        default="all",
        help="Mode: 'results' = only produce JSON results, 'plots' = only generate plots from existing JSONs, 'all' = run both (default).",
    )
    return p.parse_args()


def main():
    args = parse()
    outdir = Path(args.outdir)
    images = Path(args.images)
    allowed_bytes = int(args.allowed_mem_gb * 1024**3)
    max_elements = int(allowed_bytes / (BYTES_PER_ELEMENT * float(args.mem_factor)))
    _svd_guard.max_elements = max_elements
    print("SVD guard threshold (m*m + n*n) <=", max_elements)

    mode = args.mode
    if mode in ("all", "results"):
        with MonkeyPatch(np.linalg, "svd", _svd_guard):
            _ = run_all(outdir=outdir, repeat=args.repeat, seed=args.seed)

    if mode in ("all", "plots"):
        make_plots(outdir=outdir, images=images)


if __name__ == "__main__":
    main()
