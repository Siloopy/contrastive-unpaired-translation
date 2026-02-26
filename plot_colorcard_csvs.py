#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot colorcard ΔE2000 CSVs (small-N friendly).

This script expects per-image/per-model ΔE2000 CSVs.
If a directory contains other CSVs (e.g., rank tables) without required columns,
they will be skipped with a warning.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def infer_agg_from_name(name: str) -> str:
    low = name.lower()
    if "median" in low:
        return "median"
    if "trim" in low:
        return "trim"
    if "mean" in low:
        return "mean"
    return "unknown"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["de2000", "deltae2000", "delta_e2000", "de", "deltae"]:
            col_map[c] = "de2000"
        elif cl in ["model", "method", "net", "name"]:
            col_map[c] = "model"
        elif cl in ["image", "img", "filename", "file"]:
            col_map[c] = "image"
        elif cl in ["agg", "aggregation", "agg_method"]:
            col_map[c] = "agg"

    if col_map:
        df = df.rename(columns=col_map)
    return df


def load_csvs(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(str(input_dir / "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV found in: {input_dir}")

    dfs = []
    skipped = 0

    for p in csv_paths:
        base = os.path.basename(p).lower()

        # Common non-per-image result tables: skip early by filename hint
        if "rank" in base or "spearman" in base or "summary" in base:
            print(f"[SKIP] {p} (looks like a summary/rank table)")
            skipped += 1
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] {p} (read failed: {e})")
            skipped += 1
            continue

        df = normalize_columns(df)

        # Must have these for plotting
        if "model" not in df.columns or "de2000" not in df.columns:
            print(f"[SKIP] {p} (missing required columns: "
                  f"{'model' if 'model' not in df.columns else ''} "
                  f"{'de2000' if 'de2000' not in df.columns else ''})")
            skipped += 1
            continue

        # If agg missing, infer from filename
        if "agg" not in df.columns:
            df["agg"] = infer_agg_from_name(base)

        # If image missing, synthesize
        if "image" not in df.columns:
            df["image"] = [f"img_{i:03d}" for i in range(len(df))]

        # Clean types
        df["agg"] = df["agg"].astype(str).str.lower()
        df["model"] = df["model"].astype(str)
        df["image"] = df["image"].astype(str)
        df["de2000"] = pd.to_numeric(df["de2000"], errors="coerce")
        df = df.dropna(subset=["de2000"])

        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            f"No valid per-image CSV found in {input_dir}. "
            f"Skipped {skipped} files."
        )

    all_df = pd.concat(dfs, ignore_index=True)

    known_aggs = {"mean", "median", "trim"}
    if all_df["agg"].isin(known_aggs).any():
        all_df = all_df[all_df["agg"].isin(known_aggs)].copy()

    print(f"[INFO] Loaded {len(dfs)} CSV(s), skipped {skipped}, rows={len(all_df)}")
    return all_df


def t_critical_95(n: int) -> float:
    t_table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
        9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
        15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
    }
    if n <= 1:
        return np.nan
    return t_table.get(n, 1.96)


def savefig(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_summary_mean_ci(df: pd.DataFrame, out_dir: Path):
    models = sorted(df["model"].unique().tolist())
    aggs = [a for a in ["mean", "median", "trim"] if a in set(df["agg"].unique())]
    if not models or not aggs:
        return

    stats = []
    for m in models:
        for a in aggs:
            sub = df[(df["model"] == m) & (df["agg"] == a)]["de2000"].values
            n = len(sub)
            if n == 0:
                continue
            mu = float(np.mean(sub))
            sd = float(np.std(sub, ddof=1)) if n > 1 else 0.0
            t = t_critical_95(n)
            ci = t * sd / np.sqrt(n) if (n > 1 and np.isfinite(t)) else 0.0
            stats.append((m, a, n, mu, sd, ci))

    if not stats:
        return

    stat_df = pd.DataFrame(stats, columns=["model", "agg", "n", "mean", "std", "ci95"])

    plt.figure(figsize=(10, 4.8))
    x_base = np.arange(len(models))
    width = 0.22 if len(aggs) >= 3 else 0.3
    offsets = np.linspace(-width, width, len(aggs))

    rng = np.random.default_rng(0)  # deterministic jitter

    for i, a in enumerate(aggs):
        y, e = [], []
        for m in models:
            row = stat_df[(stat_df["model"] == m) & (stat_df["agg"] == a)]
            if len(row) == 0:
                y.append(np.nan)
                e.append(0.0)
            else:
                y.append(row["mean"].values[0])
                e.append(row["ci95"].values[0])

        xs = x_base + offsets[i]
        plt.bar(xs, y, width=width, label=a)
        plt.errorbar(xs, y, yerr=e, fmt="none", capsize=3)

        for j, m in enumerate(models):
            sub = df[(df["model"] == m) & (df["agg"] == a)]["de2000"].values
            if len(sub) == 0:
                continue
            jitter = (rng.random(len(sub)) - 0.5) * width * 0.45
            plt.scatter(np.full_like(sub, xs[j]) + jitter, sub, s=20, alpha=0.75)

    plt.xticks(x_base, models)
    plt.ylabel("ΔE2000 (mean ± 95% CI; dots = per-image)")
    plt.title("Colorcard ΔE2000 by model and aggregation")
    plt.legend(loc="best")
    savefig(out_dir / "summary_mean_ci.png")


def plot_strip_per_agg(df: pd.DataFrame, out_dir: Path):
    models = sorted(df["model"].unique().tolist())
    aggs = [a for a in ["mean", "median", "trim"] if a in set(df["agg"].unique())]
    if not models or not aggs:
        return

    rng = np.random.default_rng(0)

    for a in aggs:
        sub = df[df["agg"] == a].copy()
        plt.figure(figsize=(8.8, 4.6))
        x = np.arange(len(models))

        for i, m in enumerate(models):
            vals = sub[sub["model"] == m]["de2000"].values
            if len(vals) == 0:
                continue
            jitter = (rng.random(len(vals)) - 0.5) * 0.18
            plt.scatter(np.full_like(vals, x[i]) + jitter, vals, s=35, alpha=0.85)
            plt.hlines(np.mean(vals), x[i] - 0.22, x[i] + 0.22, linewidth=2)

        plt.xticks(x, models)
        plt.ylabel("ΔE2000 (per-image)")
        plt.title(f"Per-image ΔE2000 (agg={a})  |  dots=images, line=mean")
        savefig(out_dir / f"strip_per_agg_{a}.png")


def plot_pairwise_scatter(df: pd.DataFrame, out_dir: Path):
    wide = df.pivot_table(index=["model", "image"], columns="agg", values="de2000", aggfunc="mean")
    aggs_present = [a for a in ["mean", "median", "trim"] if a in wide.columns]
    if len(aggs_present) < 2:
        return

    pairs = []
    for i in range(len(aggs_present)):
        for j in range(i + 1, len(aggs_present)):
            pairs.append((aggs_present[i], aggs_present[j]))

    plt.figure(figsize=(5.2 * len(pairs), 4.8))
    for k, (a1, a2) in enumerate(pairs, start=1):
        ax = plt.subplot(1, len(pairs), k)
        d = wide[[a1, a2]].dropna()
        x = d[a1].values
        y = d[a2].values
        ax.scatter(x, y, s=30, alpha=0.8)

        mn = float(np.min([x.min(), y.min()]))
        mx = float(np.max([x.max(), y.max()]))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

        mae = float(np.mean(np.abs(x - y))) if len(x) else np.nan
        ax.set_xlabel(a1)
        ax.set_ylabel(a2)
        ax.set_title(f"{a1} vs {a2}\nMAE={mae:.3f}")

    plt.suptitle("Aggregation consistency (dot = model-image)", y=1.02)
    savefig(out_dir / "agg_scatter_pairwise.png")


def plot_agg_delta(df: pd.DataFrame, out_dir: Path):
    wide = df.pivot_table(index=["model", "image"], columns="agg", values="de2000", aggfunc="mean")
    cols = [c for c in ["mean", "median", "trim"] if c in wide.columns]
    if len(cols) < 2:
        return

    deltas = []
    for m in wide.index.get_level_values(0).unique():
        sub = wide.loc[m]

        def add_delta(c1: str, c2: str, name: str):
            if c1 in cols and c2 in cols:
                d = (sub[c1] - sub[c2]).abs().dropna()
                for v in d.values:
                    deltas.append((m, name, float(v)))

        add_delta("mean", "median", "abs(mean-median)")
        add_delta("mean", "trim", "abs(mean-trim)")
        add_delta("median", "trim", "abs(median-trim)")

    if not deltas:
        return

    ddf = pd.DataFrame(deltas, columns=["model", "delta_type", "abs_delta"])
    models = sorted(ddf["model"].unique().tolist())
    delta_types = sorted(ddf["delta_type"].unique().tolist())

    plt.figure(figsize=(10, 4.8))
    x_base = np.arange(len(models))
    width = 0.22 if len(delta_types) >= 3 else 0.3
    offsets = np.linspace(-width, width, len(delta_types))

    for i, dt in enumerate(delta_types):
        y = []
        for m in models:
            vals = ddf[(ddf["model"] == m) & (ddf["delta_type"] == dt)]["abs_delta"].values
            y.append(float(np.mean(vals)) if len(vals) else np.nan)
        plt.bar(x_base + offsets[i], y, width=width, label=dt)

    plt.xticks(x_base, models)
    plt.ylabel("Mean absolute ΔE difference")
    plt.title("Aggregation sensitivity (lower = more robust)")
    plt.legend(loc="best")
    savefig(out_dir / "agg_delta_bar.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="color_eval_sweep_out",
                    help="Directory containing per-image colorcard eval CSVs")
    ap.add_argument("--out_dir", type=str, default="color_eval_plots",
                    help="Directory to save plots")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(input_dir)

    plot_summary_mean_ci(df, out_dir)
    plot_strip_per_agg(df, out_dir)
    plot_pairwise_scatter(df, out_dir)
    plot_agg_delta(df, out_dir)

    print("[SAVED]")
    print(f"  {out_dir}/summary_mean_ci.png")
    for a in ["mean", "median", "trim"]:
        if a in set(df["agg"].unique()):
            print(f"  {out_dir}/strip_per_agg_{a}.png")
    print(f"  {out_dir}/agg_scatter_pairwise.png")
    print(f"  {out_dir}/agg_delta_bar.png")


if __name__ == "__main__":
    main()
