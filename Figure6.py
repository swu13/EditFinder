#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 pipeline (A/B/D/E/F)

[Figure to output mapping]
    Figure 6A: Correlation of eQTL effect size (beta) vs editing score change
               (diff), restricted to |diff| > tau
    Figure 6B: Boxplot of beta for sites with diff < -tau vs diff > +tau
    Figure 6D: ACER3 expression across editing level groups (LGG only)
    Figure 6E: ACER3 expression in TCGA-LGG Primary Tumor vs GTEx Brain Cortex
    Figure 6F: Kaplan-Meier curves for ACER3 high/low in TCGA-LGG
               (OS and DFS)

[External data (must be prepared in advance; fill paths in CONFIG['input'])]
    6A/6B:
        gpedit_snp_effects_dbsnp_matched.tsv : per-SNP eQTL + editing effect table
    6D:
        TCGAdata/<CANCER>/*HiSeqV2*.gz       : TCGA mRNA matrix (rows = genes)
        TCGAEditing/<CANCER>_Editing.txt     : TCGA editing matrix (rows = sites)
    6E/6F:
        TCGAdata/TcgaTargetGtex_rsem_gene_tpm.gz
        TCGAdata/TcgaTargetGTEX_phenotype.txt.gz
        TCGAdata/gencode.v23.annotation.gene.probemap
        TCGAdata/LGG/LGG_survival.txt
        GPEx/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt
        R packages: ggplot2, dplyr, data.table, survival, survminer

[Figure to file mapping]
    Figure 6A: gpedit/plots/diff_vs_beta_absdiff_gt_tau{tau}.png
    Figure 6B: gpedit/plots/beta_box_vs0_diff_tau{tau}.png
    Figure 6D: gpedit/mechanism/TCGAdata/LGG/editing_ACER3_output/LGG_editing_vs_ACER3_boxplot.png
    Figure 6E: gpedit/mechanism/TCGAdata/LGG/ACER3_GEPIA2_matched_output/ACER3_LGG_TCGA_vs_GTEx_Cortex_boxplot.png
    Figure 6F: gpedit/mechanism/TCGAdata/LGG/ACER3_GEPIA2_matched_output/ACER3_LGG_KM_OS.png
               gpedit/mechanism/TCGAdata/LGG/ACER3_GEPIA2_matched_output/ACER3_LGG_KM_DFS.png
"""

import os
import sys
import glob
import tempfile
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG  (replace every path below with your own before running)
# ============================================================================
CONFIG = {
    "input": {
        "fig6a_6b": {
            "in_tsv":    "/path/to/gpedit/gpedit_snp_effects_dbsnp_matched.tsv",
            "out_dir":   "/path/to/gpedit",
        },
        "fig6d": {
            "base_dir":    "/path/to/gpedit/mechanism/TCGAdata",
            "editing_dir": "/path/to/TCGAEditing",
            "gene":        "ACER3",
            "target_site": "chr11_77023912_+",
            "cancer":      "LGG",
            "min_n_plot":  10,
            "min_n_corr":  10,
        },
        "fig6e_6f": {
            "tcga_dir":     "/path/to/gpedit/mechanism/TCGAdata",
            "gtex_dir":     "/path/to/gpedit/mechanism/GPEx",
            "r_lib_path":   "",
            "gene":         "ACER3",
        },
    },
}


# ============================================================================
# Paths
# ============================================================================
GEPEDIT_OUT_DIR   = CONFIG["input"]["fig6a_6b"]["out_dir"]
GEPEDIT_IN_TSV    = CONFIG["input"]["fig6a_6b"]["in_tsv"]
GEPEDIT_DEDUP_TSV = os.path.join(GEPEDIT_OUT_DIR, "gpedit_snp_effects_dbsnp_dedup.tsv")
GEPEDIT_PLOT_DIR  = os.path.join(GEPEDIT_OUT_DIR, "plots")
GEPEDIT_SUMMARY   = os.path.join(GEPEDIT_OUT_DIR, "analysis_summary.tsv")
os.makedirs(GEPEDIT_PLOT_DIR, exist_ok=True)

rcParams["font.family"]        = "DejaVu Sans"
rcParams["axes.linewidth"]     = 1.0
rcParams["savefig.dpi"]        = 300
rcParams["savefig.bbox"]       = "tight"
rcParams["axes.unicode_minus"] = False


# ============================================================================
# Common helpers
# ============================================================================
def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = NA"
    if p < 1e-300:
        return "p < 1e-300"
    if p < 1e-4:
        return f"p = {p:.2e}"
    return f"p = {p:.4f}"


def corr_stats(x, y):
    if len(x) < 3:
        return dict(pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan, n=len(x))
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return dict(pearson_r=pr, pearson_p=pp,
                spearman_r=sr, spearman_p=sp, n=len(x))


def bootstrap_ci(x, y, method="spearman", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            if method == "spearman":
                r, _ = stats.spearmanr(x[idx], y[idx])
            else:
                r, _ = stats.pearsonr(x[idx], y[idx])
            if np.isfinite(r):
                vals.append(r)
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan, np.nan
    vals = np.asarray(vals)
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
            float((vals > 0).mean()))


def describe(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    q1=np.nan, q3=np.nan, iqr=np.nan,
                    vmin=np.nan, vmax=np.nan)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                vmin=float(v.min()), vmax=float(v.max()))


def mannwhitney(up, down):
    x = down["beta"].to_numpy()
    y = up["beta"].to_numpy()
    if len(x) < 3 or len(y) < 3:
        return dict(U=np.nan, p=np.nan)
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return dict(U=float(U), p=float(p))


def _robust_ylim(v, mode='iqr', pad_ratio=0.08, manual_range=None):
    if manual_range is not None:
        return manual_range
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    if mode == 'full':
        lo, hi = float(v.min()), float(v.max())
    elif mode == 'robust':
        q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
        lo = max(float(v.min()), q1 - 3 * iqr)
        hi = min(float(v.max()), q3 + 3 * iqr)
    elif mode == 'iqr':
        q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
        lo = max(float(v.min()), q1 - 1.5 * iqr)
        hi = min(float(v.max()), q3 + 1.5 * iqr)
    else:
        if len(v) >= 50:
            lo, hi = np.percentile(v, [2, 98]); lo, hi = float(lo), float(hi)
        else:
            lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi) * 0.1, 1e-3)
    pad = span * pad_ratio
    return lo - pad, hi + pad


def stratified_by_absdiff(df):
    if len(df) < 40:
        print("  too few rows for stratification, skip")
        return None
    df = df.copy()
    df["absdiff_q"] = pd.qcut(df["diff"].abs(), q=4,
                              labels=["Q1(weakest)", "Q2", "Q3", "Q4(strongest)"])
    print(f"\n  [stratified by |diff| quartile]")
    rows = []
    for b, sub in df.groupby("absdiff_q", observed=True):
        if len(sub) < 10:
            print(f"    {b:14s}: n={len(sub):3d}  (too few, skip)")
            continue
        rho, p = stats.spearmanr(sub["beta"], sub["diff"])
        print(f"    {b:14s}: n={len(sub):3d}  rho={rho:+.4f} (p={p:.4g})  "
              f"|diff| median={sub['diff'].abs().median():.4f}")
        rows.append(dict(analysis=f"strat_{b}", n=len(sub),
                         spearman_rho=rho, spearman_p=p))
    return rows


# ============================================================================
# Dedup / loading (Figure 6A/6B)
# ============================================================================
def _aggregate_beta(df, beta_agg):
    agg_spec = dict(
        beta=("beta", beta_agg),
        pvalue=("pvalue", "min"),
        editscore=("editscore", "first"),
        n_beta=("beta", "size"),
    )
    for col in ["rsid", "gene", "gene1"]:
        if col in df.columns:
            agg_spec[col] = (col, "first")
    return df.groupby(["chrom", "edit_start", "strand", "dis"],
                      as_index=False).agg(**agg_spec)


def _extract_unique_diffs(df):
    return (df[["chrom", "edit_start", "strand", "dis", "snptype",
                "modifiedscore", "diff"]]
            .drop_duplicates(["chrom", "edit_start", "strand", "dis", "snptype"]))


def dedup_rule(merged):
    merged = merged.copy()
    merged["absdiff"] = merged["diff"].abs()
    merged["absbeta"] = merged["beta"].abs()
    merged["sign_match"] = np.sign(merged["diff"]) == np.sign(merged["beta"])
    beta_med = merged["absbeta"].median()
    print(f"  [rule] |beta| threshold (global median) = {beta_med:.4f}")
    print(f"  [rule] sign_match rate before selection = "
          f"{merged['sign_match'].mean()*100:.1f}%")
    records = []
    n_used_same_dir = 0
    n_used_fallback = 0
    for key, sub in merged.groupby(
            ["chrom", "edit_start", "strand", "dis"], sort=False):
        same_dir = sub[sub["sign_match"]]
        if len(same_dir) > 0:
            pool = same_dir
            n_used_same_dir += 1
        else:
            pool = sub
            n_used_fallback += 1
        abs_beta = pool["absbeta"].iloc[0]
        if abs_beta >= beta_med:
            idx = pool["absdiff"].idxmax()
        else:
            idx = pool["absdiff"].idxmin()
        records.append(pool.loc[idx].to_dict())
    dedup = pd.DataFrame(records).drop(columns=["absdiff", "absbeta"])
    print(f"  [rule] sites: {len(dedup)}")
    print(f"  [rule] used same-direction alt: {n_used_same_dir}")
    print(f"  [rule] fallback (no same-direction): {n_used_fallback}")
    print(f"  [rule] sign_match rate after selection = "
          f"{dedup['sign_match'].mean()*100:.1f}%")
    return dedup


def dedup_maxdiff(merged):
    merged = merged.copy()
    merged["absdiff"] = merged["diff"].abs()
    merged["sign_match"] = np.sign(merged["diff"]) == np.sign(merged["beta"])
    dedup = (merged.sort_values(["absdiff", "dis"],
                                ascending=[False, True], kind="stable")
                    .drop_duplicates(["chrom", "edit_start", "strand", "dis"],
                                     keep="first")
                    .drop(columns=["absdiff"])
                    .reset_index(drop=True))
    print(f"  [maxdiff] sites: {len(dedup)}")
    return dedup


def dedup_random(merged, seed=42):
    rng = np.random.default_rng(seed)
    merged = merged.copy()
    merged["sign_match"] = np.sign(merged["diff"]) == np.sign(merged["beta"])
    merged["_rand"] = rng.random(len(merged))
    dedup = (merged.sort_values(["chrom", "edit_start", "strand", "dis", "_rand"])
                    .drop_duplicates(["chrom", "edit_start", "strand", "dis"],
                                     keep="first")
                    .drop(columns=["_rand"])
                    .reset_index(drop=True))
    print(f"  [random] sites: {len(dedup)}")
    return dedup


def dedup_mean(merged):
    merged = merged.copy()
    dedup = (merged.groupby(["chrom", "edit_start", "strand", "dis"],
                            as_index=False)
                    .agg(beta=("beta", "first"),
                         pvalue=("pvalue", "min"),
                         editscore=("editscore", "first"),
                         modifiedscore=("modifiedscore", "mean"),
                         diff=("diff", "mean"),
                         snptype=("snptype", "first"),
                         rsid=("rsid", "first"),
                         gene=("gene", "first"),
                         gene1=("gene1", "first"),
                         n_beta=("n_beta", "first"),
                         n_alt=("snptype", "nunique")))
    dedup["sign_match"] = np.sign(dedup["diff"]) == np.sign(dedup["beta"])
    print(f"  [mean] sites: {len(dedup)}")
    return dedup


def load_and_dedup(beta_agg="median", dedup_mode="rule", seed=42):
    df = pd.read_csv(GEPEDIT_IN_TSV, sep="\t")
    need = ["chrom", "edit_start", "strand", "dis",
            "editscore", "modifiedscore", "diff",
            "beta", "pvalue", "snptype"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    df = df.dropna(subset=["diff", "beta",
                           "modifiedscore", "editscore"]).copy()
    print(f"Raw rows                          : {len(df):,}")
    beta_agg_df = _aggregate_beta(df, beta_agg)
    print(f"SNP positions (pos, dis)          : {len(beta_agg_df):,}  "
          f"(beta = {beta_agg})")
    snp_diff = _extract_unique_diffs(df)
    print(f"Unique (pos, dis, snptype)        : {len(snp_diff):,}")
    merged = snp_diff.merge(beta_agg_df,
                            on=["chrom", "edit_start", "strand", "dis"],
                            how="inner")
    n_alt = merged.groupby(
        ["chrom", "edit_start", "strand", "dis"])["snptype"].nunique()
    print(f"  alt distribution: {n_alt.value_counts().sort_index().to_dict()}")
    print(f"\nDedup mode: {dedup_mode}")
    if dedup_mode == "rule":
        dedup = dedup_rule(merged)
    elif dedup_mode == "maxdiff":
        dedup = dedup_maxdiff(merged)
    elif dedup_mode == "random":
        dedup = dedup_random(merged, seed=seed)
    elif dedup_mode == "mean":
        dedup = dedup_mean(merged)
    else:
        raise ValueError(f"unknown dedup_mode: {dedup_mode}")
    col_order = ["chrom", "edit_start", "strand", "dis",
                 "editscore", "modifiedscore", "diff",
                 "beta", "pvalue", "snptype",
                 "rsid", "gene", "gene1", "n_beta", "sign_match"]
    col_order = [c for c in col_order if c in dedup.columns]
    dedup = dedup[col_order].reset_index(drop=True)
    dedup.to_csv(GEPEDIT_DEDUP_TSV, sep="\t", index=False)
    print(f"  -> {GEPEDIT_DEDUP_TSV}  ({len(dedup)} rows)")
    if "n_beta" in dedup.columns:
        print(f"  sites with n_beta > 1: {(dedup['n_beta'] > 1).sum():,}")
        print(f"  n_beta distribution : "
              f"{dedup['n_beta'].value_counts().sort_index().to_dict()}")
    if "sign_match" in dedup.columns:
        print(f"  sign_match rate in final dedup: "
              f"{dedup['sign_match'].mean()*100:.1f}%")
    return dedup


def filter_by_abs_beta(df, cutoff, keep="above"):
    if cutoff is None or cutoff <= 0:
        print(f"\n|beta| filter: disabled (cutoff={cutoff})")
        return df.copy()
    m = df["beta"].abs() >= cutoff
    df_f = df[m].copy() if keep == "above" else df[~m].copy()
    op = ">=" if keep == "above" else "<"
    print(f"\n|beta| {op} {cutoff} filter (keep={keep}):")
    print(f"  before : {len(df):>4d}")
    print(f"  after  : {len(df_f):>4d}")
    print(f"  removed: {len(df) - len(df_f):>4d}")
    return df_f


def split_by_tau(df, tau):
    up   = df[df["diff"] >  tau].copy()
    down = df[df["diff"] < -tau].copy()
    print(f"\nSplit by |diff| > {tau}:")
    print(f"  diff >  +{tau}: n = {len(up):>4d}")
    print(f"  diff <  -{tau}: n = {len(down):>4d}")
    print(f"  |diff| <= {tau} (excluded): n = {len(df) - len(up) - len(down):>4d}")
    return up, down


# ============================================================================
# Figure 6A / 6B
# ============================================================================
def plot_beta_box_vs_zero(up, down, tau,
                          ylim_mode='iqr', show_fliers=False,
                          manual_range=None):
    """Figure 6B: boxplot of beta; caption only reports Mann-Whitney p."""
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    data   = [down["beta"].to_numpy(), up["beta"].to_numpy()]
    labels = [f"diff < -{tau}\n(n={len(down)})",
              f"diff > +{tau}\n(n={len(up)})"]
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=0.5,
        showfliers=show_fliers, whis=1.5,
        medianprops=dict(color="black", lw=1.6),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="grey", markeredgecolor="none",
                        alpha=0.45),
    )
    for patch, c in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.85, zorder=0)
    all_beta = np.concatenate([d for d in data if len(d) > 0])
    ylim = _robust_ylim(all_beta, mode=ylim_mode, manual_range=manual_range)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("beta (eQTL effect size)")
    ax.set_title(f"beta distribution: diff < -{tau} vs diff > +{tau}")
    ax.grid(alpha=0.25, axis="y", lw=0.5)

    mw_beta = mannwhitney(up, down)
    # Caption: only Mann-Whitney p
    caption = f"up vs down:  Mann-Whitney p = {mw_beta['p']:.4g}"
    fig.text(0.5, -0.02, caption, ha="center", va="top",
             fontsize=9, family="monospace")

    out = os.path.join(GEPEDIT_PLOT_DIR, f"beta_box_vs0_diff_tau{tau}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")
    return mw_beta


def plot_corr_under_tau(up, down, tau):
    """Figure 6A: pooled scatter of diff vs beta, |diff| > tau."""
    both = pd.concat([down, up], ignore_index=True)
    x = both["beta"].to_numpy()
    y = both["diff"].to_numpy()
    st = corr_stats(x, y)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x, y, s=26, alpha=0.75,
               color="#4C72B0", edgecolors="white", linewidths=0.4)
    if len(both) >= 3:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept,
                color="black", lw=1.3, ls="--", alpha=0.85)
    ax.axhline(0, color="grey", lw=0.6, alpha=0.6)
    ax.axvline(0, color="grey", lw=0.6, alpha=0.6)
    ax.set_xlabel("beta (eQTL effect size)")
    ax.set_ylabel("diff = modifiedscore - editscore")
    ax.set_title(f"diff vs beta (|diff| > {tau})")
    ax.grid(alpha=0.25, lw=0.5)

    caption = (
        f"|diff| > {tau},  n = {st['n']},  "
        f"Pearson r = {st['pearson_r']:+.3f} (p = {st['pearson_p']:.4g});  "
        f"Spearman rho = {st['spearman_r']:+.3f} (p = {st['spearman_p']:.4g})"
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top",
             fontsize=9, family="monospace")
    out = os.path.join(GEPEDIT_PLOT_DIR, f"diff_vs_beta_absdiff_gt_tau{tau}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")
    return st


def run_fig6a_6b():
    print("\n" + "=" * 70)
    print("Figure 6A - |diff|>tau scatter, Figure 6B - beta boxplot")
    print("=" * 70)
    if not os.path.exists(GEPEDIT_IN_TSV):
        raise FileNotFoundError(f"input not found: {GEPEDIT_IN_TSV}")

    tau_list = [0.001]
    beta_abs_cut = 0.0
    beta_agg = "median"
    dedup_mode = "rule"
    box_ylim = "iqr"
    box_ylim_range = None
    box_fliers = False

    manual_range = None
    if box_ylim_range:
        try:
            parts = box_ylim_range.split(',')
            manual_range = (float(parts[0].strip()), float(parts[1].strip()))
            print(f"  [box] manual ylim range = {manual_range}")
        except Exception as e:
            print(f"  WARNING bad --box-ylim-range format: {e}", file=sys.stderr)
            manual_range = None

    summary_rows = []
    df = load_and_dedup(beta_agg=beta_agg, dedup_mode=dedup_mode)
    df = filter_by_abs_beta(df, beta_abs_cut, keep="above")
    if len(df) == 0:
        raise RuntimeError("no sites after filtering")

    # PRIMARY analysis
    print("\n" + "=" * 60)
    print(f"PRIMARY ANALYSIS (dedup-mode={dedup_mode})")
    print("=" * 60)
    x_full = df["beta"].to_numpy()
    y_full = df["diff"].to_numpy()
    st_main = corr_stats(x_full, y_full)
    ci_lo, ci_hi, prob_pos = bootstrap_ci(x_full, y_full, "spearman")
    ci_lo_p, ci_hi_p, _ = bootstrap_ci(x_full, y_full, "pearson")
    print(f"  n = {st_main['n']:,}")
    print(f"  Spearman rho = {st_main['spearman_r']:+.4f}  "
          f"{fmt_p(st_main['spearman_p'])}  "
          f"bootstrap 95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
          f"P(rho>0) = {prob_pos:.3f}")
    print(f"  Pearson  r   = {st_main['pearson_r']:+.4f}  "
          f"{fmt_p(st_main['pearson_p'])}  "
          f"bootstrap 95% CI = [{ci_lo_p:+.3f}, {ci_hi_p:+.3f}]")
    summary_rows.append(dict(
        analysis=f"PRIMARY_{dedup_mode}", n=st_main["n"],
        spearman_rho=st_main["spearman_r"], spearman_p=st_main["spearman_p"],
        spearman_ci_lo=ci_lo, spearman_ci_hi=ci_hi,
        pearson_r=st_main["pearson_r"], pearson_p=st_main["pearson_p"],
        pearson_ci_lo=ci_lo_p, pearson_ci_hi=ci_hi_p,
        note=f"dedup-mode={dedup_mode}"))

    print("\n  per strand:")
    for s in ["+", "-"]:
        sub = df[df["strand"] == s]
        if len(sub) >= 10:
            r2, p2 = stats.spearmanr(sub["beta"], sub["diff"])
            print(f"    strand {s}: n={len(sub):,}, rho={r2:+.4f} (p={p2:.4g})")
            summary_rows.append(dict(
                analysis=f"PRIMARY_{dedup_mode}_strand_{s}", n=len(sub),
                spearman_rho=r2, spearman_p=p2,
                spearman_ci_lo=np.nan, spearman_ci_hi=np.nan,
                pearson_r=np.nan, pearson_p=np.nan,
                pearson_ci_lo=np.nan, pearson_ci_hi=np.nan,
                note="primary per strand"))

    strat_rows = stratified_by_absdiff(df)
    if strat_rows:
        for r in strat_rows:
            summary_rows.append(dict(
                analysis=f"{dedup_mode}_{r['analysis']}", n=r["n"],
                spearman_rho=r["spearman_rho"], spearman_p=r["spearman_p"],
                spearman_ci_lo=np.nan, spearman_ci_hi=np.nan,
                pearson_r=np.nan, pearson_p=np.nan,
                pearson_ci_lo=np.nan, pearson_ci_hi=np.nan,
                note="stratified"))

    for tau in tau_list:
        print("\n" + "=" * 60)
        print(f"SENSITIVITY (selection on y = |diff| > {tau})")
        print("=" * 60)
        up, down = split_by_tau(df, tau)
        if len(up) == 0 or len(down) == 0:
            print("  one group is empty, skip.")
            continue
        mw_beta = plot_beta_box_vs_zero(
            up, down, tau,
            ylim_mode=box_ylim,
            show_fliers=box_fliers,
            manual_range=manual_range,
        )
        print(f"\n  [Mann-Whitney U between groups (up vs down)]  "
              f"U = {mw_beta['U']:.1f}, {fmt_p(mw_beta['p'])}")
        st = plot_corr_under_tau(up, down, tau)
        print(f"\n  [pooled correlation |diff| > {tau}]")
        print(f"    Pearson  r   = {st['pearson_r']:+.4f}, "
              f"{fmt_p(st['pearson_p'])}")
        print(f"    Spearman rho = {st['spearman_r']:+.4f}, "
              f"{fmt_p(st['spearman_p'])}")
        print(f"    n = {st['n']}")

    pd.DataFrame(summary_rows).to_csv(GEPEDIT_SUMMARY, sep="\t", index=False)
    print(f"\nSummary -> {GEPEDIT_SUMMARY}")
    print(f"All plots saved under: {GEPEDIT_PLOT_DIR}")


# ============================================================================
# Figure 6D  (ACER3 vs editing level in LGG only)
# ============================================================================
def _sid(x):
    return str(x)[:15]


def _find_file(pattern):
    hits = sorted(glob.glob(pattern))
    for h in hits:
        if not any(k in h.lower() for k in ["pancan", "percentile"]):
            return h
    return hits[0] if hits else None


def _spearman_report(x, y, n_min):
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < n_min:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return rho, p, n


def run_fig6d():
    print("\n" + "=" * 70)
    print("Figure 6D - ACER3 vs editing level (LGG only)")
    print("=" * 70)
    cfg = CONFIG["input"]["fig6d"]
    base_dir    = cfg["base_dir"]
    editing_dir = cfg["editing_dir"]
    gene        = cfg["gene"]
    target_site = cfg["target_site"]
    cancer      = cfg["cancer"]
    min_n_plot  = cfg["min_n_plot"]
    min_n_corr  = cfg["min_n_corr"]

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})

    cancer_dir = os.path.join(base_dir, cancer)
    out_dir    = os.path.join(cancer_dir, "editing_ACER3_output")
    os.makedirs(out_dir, exist_ok=True)

    mrna_file    = _find_file(os.path.join(cancer_dir, "*HiSeqV2*.gz"))
    editing_file = os.path.join(editing_dir, f"{cancer}_Editing.txt")
    if mrna_file is None or not os.path.exists(editing_file):
        raise FileNotFoundError(
            f"missing mRNA or editing file for {cancer}:\n"
            f"  mRNA   : {mrna_file}\n"
            f"  editing: {editing_file}")

    print(f"\n{'='*70}\nProcessing {cancer}\n{'='*70}")

    mrna = pd.read_csv(mrna_file, sep="\t", index_col=0)
    if gene not in mrna.index:
        raise KeyError(f"{gene} not in mRNA row names")
    acer3 = pd.to_numeric(mrna.loc[gene], errors="coerce")

    edit_df = pd.read_csv(editing_file, sep="\t", index_col=0)
    if target_site not in edit_df.index:
        raise KeyError(f"{target_site} not in editing matrix")
    edit = pd.to_numeric(edit_df.loc[target_site], errors="coerce").dropna()

    df_edit = pd.DataFrame({"sample_key": [_sid(s) for s in edit.index],
                            "editing": edit.values}).dropna()
    df_acer3 = pd.DataFrame({"sample_key": [_sid(s) for s in acer3.index],
                             "ACER3": acer3.values}).dropna()
    merged = df_edit.merge(df_acer3, on="sample_key", how="inner")

    n_edit  = len(df_edit)
    n_acer3 = len(df_acer3)
    n_int   = len(merged)
    print(f"  editing={n_edit}, ACER3={n_acer3}, intersect={n_int}")

    row = {
        "cancer": cancer,
        "n_editing_samples": n_edit,
        "n_ACER3_samples": n_acer3,
        "n_intersect": n_int,
        "editing_mean": np.nan, "editing_median": np.nan,
        "editing_min": np.nan, "editing_max": np.nan,
        "ACER3_mean": np.nan, "ACER3_median": np.nan,
        "spearman_rho": np.nan, "spearman_p": np.nan,
        "wilcoxon_p": np.nan,
        "low_group_n": np.nan, "high_group_n": np.nan,
        "low_ACER3_mean": np.nan, "high_ACER3_mean": np.nan,
        "direction": ""
    }

    summary_path = os.path.join(base_dir, "editing_vs_ACER3_summary.csv")

    if n_int < min_n_corr:
        print(f"  [SKIP] intersect < {min_n_corr}; descriptive stats only")
        if n_int > 0:
            row.update({
                "editing_mean": round(merged.editing.mean(), 4),
                "editing_median": round(merged.editing.median(), 4),
                "editing_min": round(merged.editing.min(), 4),
                "editing_max": round(merged.editing.max(), 4),
                "ACER3_mean": round(merged.ACER3.mean(), 4),
                "ACER3_median": round(merged.ACER3.median(), 4),
            })
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        return

    row.update({
        "editing_mean": round(merged.editing.mean(), 4),
        "editing_median": round(merged.editing.median(), 4),
        "editing_min": round(merged.editing.min(), 4),
        "editing_max": round(merged.editing.max(), 4),
        "ACER3_mean": round(merged.ACER3.mean(), 4),
        "ACER3_median": round(merged.ACER3.median(), 4),
    })

    rho, p, n = _spearman_report(merged.editing, merged.ACER3, min_n_corr)
    row["spearman_rho"] = round(rho, 3)
    row["spearman_p"] = p
    row["direction"] = "negative" if rho < 0 else "positive"
    print(f"  [Spearman] rho={rho:.3f}, p={p:.2e}, n={n}")

    merged["editing_group"] = pd.qcut(merged.editing, q=2, labels=["Low", "High"])
    lo = merged.loc[merged.editing_group == "Low", "ACER3"]
    hi = merged.loc[merged.editing_group == "High", "ACER3"]
    _, p_mw = stats.mannwhitneyu(lo, hi, alternative="two-sided")
    row["wilcoxon_p"] = p_mw
    row["low_group_n"] = len(lo)
    row["high_group_n"] = len(hi)
    row["low_ACER3_mean"] = round(lo.mean(), 4)
    row["high_ACER3_mean"] = round(hi.mean(), 4)
    print(f"  [Wilcoxon] Low(n={len(lo)}) vs High(n={len(hi)}), p={p_mw:.2e}")

    merged["cancer"] = cancer
    merged.to_csv(os.path.join(out_dir, f"{cancer}_edit_ACER3.csv"), index=False)
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    print(f"  summary -> {summary_path}")

    if n_int >= min_n_plot:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        sns.boxplot(data=merged, x="editing_group", y="ACER3", ax=ax,
                    width=0.55, showfliers=False)
        sns.stripplot(data=merged, x="editing_group", y="ACER3", ax=ax,
                      color="black", size=2.5, alpha=0.5, jitter=0.2)
        ax.set_xlabel("Editing level (median split)")
        ax.set_ylabel(f"{gene} expression (log2 RSEM)")
        ax.set_title(f"{cancer}: editing vs {gene}\n"
                     f"n={len(lo)} vs {len(hi)}; Wilcoxon p={p_mw:.2e}\n"
                     f"Spearman rho={rho:.3f}, p={p:.2e}, n={n}")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{cancer}_editing_vs_ACER3_boxplot.png")
        plt.savefig(out_png)
        plt.close()
        print(f"  figure -> {out_png}")


# ============================================================================
# Figure 6E / 6F  (R code embedded; executed via Rscript)
# ============================================================================
FIG6E_F_R_CODE = r"""
.libPaths(c(Sys.getenv("R_LIB_PATH", unset = ""), .libPaths()))
.libPaths(.libPaths()[nzchar(.libPaths())])

pkgs <- c("ggplot2","dplyr","data.table","survival","survminer")
miss <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(miss) > 0) {
  stop("Missing R packages: ", paste(miss, collapse = ", "),
       "\nPlease install them first, e.g.:\n",
       'Rscript -e \'install.packages(c("ggplot2","dplyr","data.table",',
       '"survival","survminer"))\'')
}
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(data.table)
  library(survival); library(survminer)
})

theme_no_grid <- theme_bw(base_size = 11) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

tcga_dir   <- Sys.getenv("TCGA_DIR",   unset = "")
gtex_dir   <- Sys.getenv("GTEX_DIR",   unset = "")
gene       <- Sys.getenv("TARGET_GENE", unset = "ACER3")
if (!nzchar(tcga_dir) || !nzchar(gtex_dir)) {
  stop("Environment variables TCGA_DIR / GTEX_DIR must be set")
}

lgg_dir    <- file.path(tcga_dir, "LGG")
out_dir    <- file.path(lgg_dir, "ACER3_GEPIA2_matched_output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

toil_joint_file <- file.path(tcga_dir, "TcgaTargetGtex_rsem_gene_tpm.gz")
toil_pheno_file <- file.path(tcga_dir, "TcgaTargetGTEX_phenotype.txt.gz")
probemap_file   <- file.path(tcga_dir, "gencode.v23.annotation.gene.probemap")
tcga_surv_file  <- file.path(lgg_dir,  "LGG_survival.txt")
gtex_attr_file  <- file.path(gtex_dir, "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt")

for (f in c(toil_joint_file, toil_pheno_file, probemap_file,
            tcga_surv_file, gtex_attr_file)) {
  if (!file.exists(f)) stop("File not found: ", f)
}

cat("Reading Toil phenotype ...\n")
pheno <- fread(cmd = paste("zcat", shQuote(toil_pheno_file)),
               sep = "\t", header = TRUE, data.table = FALSE)

lgg_mask <- pheno[["_study"]] == "TCGA" &
            pheno[["detailed_category"]] == "Brain Lower Grade Glioma" &
            pheno[["_sample_type"]] == "Primary Tumor"
lgg_samples <- pheno$sample[lgg_mask]
cat("LGG Primary Tumor samples:", length(lgg_samples), "\n")

cat("\nReading GTEx sample attributes ...\n")
gtex_attr <- fread(gtex_attr_file, sep = "\t", header = TRUE, data.table = FALSE)
cat("SMTSD Brain values:\n")
print(unique(gtex_attr$SMTSD[grepl("Brain", gtex_attr$SMTSD, ignore.case = TRUE)]))

cortex_mask <- grepl("^Brain - Cortex$|^Brain - Frontal Cortex",
                     gtex_attr$SMTSD, ignore.case = TRUE)
gtex_cortex_ids <- gtex_attr$SAMPID[cortex_mask]
cat("GTEx cortex samples in annotation:", length(gtex_cortex_ids), "\n")
cat("First 5 cortex SAMPIDs:", paste(head(gtex_cortex_ids, 5), collapse = " | "), "\n")
cat("Cortex SAMPID prefix distribution:\n")
print(table(sub("-.*$", "", gtex_cortex_ids)))

gtex_in_toil <- pheno$sample[pheno[["_study"]] == "GTEX"]
cat("GTEx samples in Toil:", length(gtex_in_toil), "\n")
cat("First 5 Toil GTEx columns:", paste(head(gtex_in_toil, 5), collapse = " | "), "\n")
cat("Toil GTEx prefix distribution:\n")
print(table(sub("-.*$", "", gtex_in_toil)))

gtex_cortex_samples <- intersect(gtex_cortex_ids, gtex_in_toil)
cat("Cortex samples matched by full ID:", length(gtex_cortex_samples), "\n")

if (length(gtex_cortex_samples) < 50) {
  cat("Full ID match insufficient, trying donor prefix matching ...\n")
  donor_prefix <- function(x) paste(strsplit(x, "-")[[1]][1:2], collapse = "-")
  cortex_donors <- unique(sapply(gtex_cortex_ids, donor_prefix))
  toil_donors   <- sapply(pheno$sample[pheno[["_study"]] == "GTEX"], donor_prefix)
  cat("Cortex donors:", length(cortex_donors), "\n")
  cat("Toil samples matched by donors:", sum(toil_donors %in% cortex_donors), "\n")
  gtex_cortex_samples <- pheno$sample[pheno[["_study"]] == "GTEX" &
                                      toil_donors %in% cortex_donors]
  cat("Cortex samples after loose matching:", length(gtex_cortex_samples), "\n")
}

if (length(gtex_cortex_samples) < 10) {
  stop("Too few cortex samples in Toil. Check SAMPID prefixes above ",
       "and adjust matching strategy.")
}

cat("\nReading gencode.v23 probemap ...\n")
probe_map <- fread(probemap_file, sep = "\t", header = TRUE, data.table = FALSE)
gene_rows <- probe_map[probe_map$gene == gene, , drop = FALSE]
if (nrow(gene_rows) == 0) stop("Cannot find ", gene, " in probemap")
gene_id <- gene_rows$id[1]
gene_id_prefix <- sub("\\..*$", "", gene_id)
cat("Using Ensembl ID:", gene_id, "\n")

cat("\nExtracting", gene, "row from Toil matrix ...\n")
awk_cmd <- sprintf(
  "zcat %s | awk -F'\\t' 'NR==1 || $1 ~ /^%s/'",
  shQuote(toil_joint_file), gene_id_prefix
)
toil_gene <- fread(cmd = awk_cmd, sep = "\t", header = TRUE, data.table = FALSE)
if (nrow(toil_gene) == 0) stop("Not found: ", gene_id_prefix)
if (nrow(toil_gene) > 1) toil_gene <- toil_gene[1, , drop = FALSE]

toil_ids <- colnames(toil_gene)[-1]
lgg_cols  <- which(toil_ids %in% lgg_samples)
gtex_cols <- which(toil_ids %in% gtex_cortex_samples)
cat("Matched LGG samples:", length(lgg_cols), "\n")
cat("Matched GTEx Cortex samples:", length(gtex_cols), "\n")
if (length(lgg_cols) < 50)  stop("Too few LGG samples matched")
if (length(gtex_cols) < 10) stop("Too few GTEx Cortex samples matched")

to_log2_plus1 <- function(x) {
  tpm <- 2^x - 0.001
  tpm[tpm < 0] <- 0
  log2(tpm + 1)
}
gene_tcga <- to_log2_plus1(as.numeric(toil_gene[1, 1 + lgg_cols]))
gene_gtex <- to_log2_plus1(as.numeric(toil_gene[1, 1 + gtex_cols]))
gene_tcga <- gene_tcga[is.finite(gene_tcga)]
gene_gtex <- gene_gtex[is.finite(gene_gtex)]

cat("TCGA-LGG", gene, "(log2 TPM+1) range:",
    round(min(gene_tcga), 2), "-", round(max(gene_tcga), 2), "\n")
cat("GTEx Cortex", gene, "(log2 TPM+1) range:",
    round(min(gene_gtex), 2), "-", round(max(gene_gtex), 2), "\n")

# ---------- Figure 6E: boxplot TCGA-LGG vs GTEx Cortex ----------
df_plot <- data.frame(
  expression = c(gene_tcga, gene_gtex),
  cohort = factor(c(rep("TCGA-LGG\n(Primary Tumor)", length(gene_tcga)),
                    rep("GTEx Brain Cortex\n(Normal)", length(gene_gtex))),
                  levels = c("TCGA-LGG\n(Primary Tumor)",
                             "GTEx Brain Cortex\n(Normal)"))
)

log2fc   <- mean(gene_tcga) - mean(gene_gtex)
p_wilcox <- wilcox.test(gene_tcga, gene_gtex)$p.value
p_anova  <- summary(aov(expression ~ cohort, data = df_plot))[[1]][["Pr(>F)"]][1]

star <- if (p_wilcox >= 0.01) {
  "ns"
} else if (abs(log2fc) >= 2) {
  "***"
} else if (abs(log2fc) >= 1.5) {
  "**"
} else if (abs(log2fc) >= 1) {
  "*"
} else {
  "ns"
}

summary_expr <- data.frame(
  cohort = c("TCGA-LGG Primary Tumor", "GTEx Brain Cortex"),
  n = c(length(gene_tcga), length(gene_gtex)),
  mean_log2 = c(mean(gene_tcga), mean(gene_gtex)),
  median_log2 = c(median(gene_tcga), median(gene_gtex)),
  sd_log2 = c(sd(gene_tcga), sd(gene_gtex)),
  log2FC = c(log2fc, NA),
  wilcoxon_p = c(p_wilcox, NA),
  anova_p = c(p_anova, NA),
  star = c(star, NA)
)
write.csv(summary_expr,
          file.path(out_dir, "ACER3_expression_summary.csv"),
          row.names = FALSE)
cat("\nExpression summary:\n"); print(summary_expr)
cat("log2FC =", round(log2fc, 3),
    "; Wilcoxon p =", format(p_wilcox, digits = 3),
    "; ANOVA p =", format(p_anova, digits = 3),
    "; star =", star, "\n")

y_max  <- max(df_plot$expression, na.rm = TRUE)
y_star <- y_max * 1.05
x_star <- 1.5

p_box <- ggplot(df_plot, aes(x = cohort, y = expression, fill = cohort)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.85) +
  geom_jitter(width = 0.18, size = 0.6, alpha = 0.35, color = "black") +
  scale_fill_manual(values = c("#d7301f", "#2c7fb8")) +
  annotate("text", x = x_star, y = y_star, label = star,
           size = 6, fontface = "bold") +
  labs(x = NULL, y = "Log2(TPM + 1) of ACER3",
       title = "ACER3 in LGG: TCGA vs GTEx Brain Cortex",
       subtitle = sprintf("n = %d vs %d; log2FC = %.2f; Wilcoxon p = %.2e",
                          length(gene_tcga), length(gene_gtex),
                          log2fc, p_wilcox)) +
  theme_no_grid +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))
ggsave(file.path(out_dir, "ACER3_LGG_TCGA_vs_GTEx_Cortex_boxplot.png"),
       p_box, width = 5, height = 4.5, dpi = 300)

# ---------- Figure 6F: KM survival (OS + DFS via PFI) ----------
cat("\nSurvival analysis ...\n")
surv <- fread(tcga_surv_file, sep = "\t", header = TRUE, data.table = FALSE)
if (!all(c("sample", "OS", "OS.time") %in% colnames(surv))) {
  stop("survival file missing sample / OS / OS.time columns")
}
surv$sample <- as.character(surv$sample)

df_surv <- data.frame(sample = toil_ids[lgg_cols],
                      ACER3 = as.numeric(toil_gene[1, 1 + lgg_cols]))
df_surv$sample_key <- substr(df_surv$sample, 1, 15)
surv$sample_key <- substr(surv$sample, 1, 15)

if (anyDuplicated(surv$sample_key)) stop("duplicate keys in survival file")
if (anyDuplicated(df_surv$sample_key)) {
  df_surv <- df_surv %>%
    group_by(sample_key) %>%
    slice_max(ACER3, n = 1, with_ties = FALSE) %>%
    ungroup()
}

keep_cols <- intersect(c("sample_key", "OS", "OS.time",
                         "DFI", "DFI.time", "PFI", "PFI.time"),
                       colnames(surv))
df_surv <- merge(df_surv, surv[, keep_cols, drop = FALSE], by = "sample_key")

plot_km_gepia <- function(d, ep_name, event_col, time_col,
                          out_file, ylab, title) {
  d$E <- as.numeric(d[[event_col]])
  d$T <- as.numeric(d[[time_col]])
  d <- d[is.finite(d$ACER3) & is.finite(d$E) & is.finite(d$T) & d$T > 0, ]
  if (nrow(d) < 20) {
    cat("  [SKIP]", ep_name, "valid samples < 20\n"); return(NULL)
  }
  if (!all(d$E %in% c(0, 1))) stop(ep_name, " event column is not 0/1")

  cut_val <- median(d$ACER3, na.rm = TRUE)
  d$group <- factor(ifelse(d$ACER3 > cut_val, "High", "Low"),
                    levels = c("Low", "High"))
  n_low  <- sum(d$group == "Low"); n_high <- sum(d$group == "High")
  if (length(unique(d$group)) < 2) return(NULL)

  fit <- survfit(Surv(T, E) ~ group, data = d)
  lr  <- survdiff(Surv(T, E) ~ group, data = d)
  lr_p <- 1 - pchisq(lr$chisq, df = 1)
  cox <- coxph(Surv(T, E) ~ group, data = d)
  cox_s <- summary(cox)
  hr   <- cox_s$conf.int[1, "exp(coef)"]
  hr_p <- cox_s$coefficients[1, "Pr(>|z|)"]

  anno <- sprintf(
    "Logrank P = %.3g\nHR(high) = %.2f\nP(HR) = %.3g\nn(high) = %d\nn(low) = %d",
    lr_p, hr, hr_p, n_high, n_low)

  p <- ggsurvplot(fit, data = d, pval = FALSE, conf.int = TRUE,
                  risk.table = FALSE,
                  palette = c("#2c7fb8", "#d7301f"),
                  legend.title = "ACER3",
                  legend.labs = c(paste0("Low (n=",  n_low,  ")"),
                                  paste0("High (n=", n_high, ")")),
                  xlab = "Time (days)", ylab = ylab, title = title,
                  ggtheme = theme_no_grid)
  p$plot <- p$plot + annotate("text", x = Inf, y = Inf, label = anno,
                              hjust = 1.05, vjust = 1.1, size = 3.2)
  png(out_file, width = 6, height = 5, units = "in", res = 300)
  print(p); dev.off()

  cat(sprintf(
    "  %s: n=%d (Low %d, High %d), events=%d, logrank p=%.3g, HR=%.2f\n",
    ep_name, nrow(d), n_low, n_high, sum(d$E == 1), lr_p, hr))

  data.frame(endpoint = ep_name, cutoff = cut_val, n_total = nrow(d),
             n_low = n_low, n_high = n_high,
             events_low  = sum(d$E[d$group == "Low"]  == 1),
             events_high = sum(d$E[d$group == "High"] == 1),
             logrank_p = lr_p, HR_high = hr,
             HR_lower95 = cox_s$conf.int[1, "lower .95"],
             HR_upper95 = cox_s$conf.int[1, "upper .95"],
             P_HR = hr_p)
}

res_os <- plot_km_gepia(df_surv, "OS", "OS", "OS.time",
                        file.path(out_dir, "ACER3_LGG_KM_OS.png"),
                        "Overall survival probability",
                        "ACER3 in TCGA-LGG: Overall Survival")

res_dfs <- NULL
if ("PFI" %in% colnames(df_surv) && "PFI.time" %in% colnames(df_surv)) {
  res_dfs <- plot_km_gepia(df_surv, "DFS (PFI)", "PFI", "PFI.time",
                           file.path(out_dir, "ACER3_LGG_KM_DFS.png"),
                           "Disease-free survival probability",
                           "ACER3 in TCGA-LGG: Disease-Free Survival")
}
if (is.null(res_dfs) && "DFI" %in% colnames(df_surv)) {
  cat("  PFI unavailable or too few samples, falling back to DFI\n")
  res_dfs <- plot_km_gepia(df_surv, "DFS (DFI)", "DFI", "DFI.time",
                           file.path(out_dir, "ACER3_LGG_KM_DFS.png"),
                           "Disease-free survival probability",
                           "ACER3 in TCGA-LGG: Disease-Free Survival")
}

surv_summary <- rbind(res_os, res_dfs)
if (!is.null(surv_summary)) {
  write.csv(surv_summary,
            file.path(out_dir, "ACER3_LGG_survival_summary.csv"),
            row.names = FALSE)
  cat("\nSurvival summary:\n"); print(surv_summary)
}

cat("\nDone. Output directory:", out_dir, "\n")
"""


def run_fig6e_6f():
    print("\n" + "=" * 70)
    print("Figure 6E - TCGA-LGG vs GTEx Cortex; Figure 6F - KM curves")
    print("=" * 70)
    cfg = CONFIG["input"]["fig6e_6f"]
    tcga_dir = cfg["tcga_dir"]
    gtex_dir = cfg["gtex_dir"]
    r_lib    = cfg["r_lib_path"]
    gene     = cfg["gene"]

    if not os.path.isdir(tcga_dir):
        raise FileNotFoundError(f"TCGA_DIR not found: {tcga_dir}")
    if not os.path.isdir(gtex_dir):
        raise FileNotFoundError(f"GTEX_DIR not found: {gtex_dir}")

    env = os.environ.copy()
    env.update({
        "TCGA_DIR":   tcga_dir,
        "GTEX_DIR":   gtex_dir,
        "R_LIB_PATH": r_lib,
        "TARGET_GENE": gene,
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(FIG6E_F_R_CODE)
        r_script_path = f.name

    try:
        print(f"Running R script: {r_script_path}")
        subprocess.run(["Rscript", r_script_path], env=env, check=True)
    finally:
        try:
            os.unlink(r_script_path)
        except OSError:
            pass


# ============================================================================
# Main
# ============================================================================
def main():

    print("=" * 70)
    print("Figure 6 pipeline - steps: all")
    print("=" * 70)

    try:
        run_fig6a_6b()
        run_fig6d()
        run_fig6e_6f()
        print("\n=== All Figure 6 panels completed ===")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()