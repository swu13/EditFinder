#!/usr/bin/env python3
"""
[Figure and output mapping]
Figure4A: figure4A_absdis_vs_median.png, A_pooled_full.tsv, A_stepA_abs_dis_trend.tsv
Figure4B: figure4B_count_ratio_combined.png, B_pos_full.tsv
Figure4C: Figure4C_dis+1.png, Figure4C_heatmap_dis+1.png, C_ref_base_stats_dis+1.tsv, E_dis1_summary_stats.csv, E_dis1_dunn_holm.csv
Figure4D: Figure4D_dis-1.png, D_ref_base_stats_dis-1.tsv
Figure4E: Figure4E_boxplot_GroupA_vs_0.png
Figure4F: Figure4F_top15_triplet_transitions.png, E_triplet_transition_summary.tsv, E_transition_pairwise_tests.tsv

[External data (prepare in advance; fill paths in CONFIG['input'])]
- parquet_a: SNP-expanded prediction parquet for A
- parquet_t: SNP-expanded prediction parquet for T
- ref_genome: reference genome FASTA
- bedtools must be available in PATH

[Figure <-> file mapping]
Figure4A -> figure4A_absdis_vs_median.png
Figure4B -> figure4B_count_ratio_combined.png
Figure4C -> Figure4C_dis+1.png, Figure4C_heatmap_dis+1.png
Figure4D -> Figure4D_dis-1.png
Figure4E -> Figure4E_boxplot_GroupA_vs_0.png
Figure4F -> Figure4F_top15_triplet_transitions.png
"""

import os
import shutil
import subprocess
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr, mannwhitneyu, norm, kruskal

try:
    import scikit_posthocs as sp
    USE_SP = True
except ImportError:
    USE_SP = False
    print("Warning: scikit-posthocs not installed.")

warnings.filterwarnings("ignore")

CONFIG = {
    "input": {
        "parquet_a": "PATH/TO/A/snp_expanded_predictions.parquet",
        "parquet_t": "PATH/TO/T/snp_expanded_with_predictions.parquet",
        "ref_genome": "PATH/TO/REFERENCE/genome.fa",
    },
    "output": {
        "fig_dir": "PATH/TO/OUTPUT/Figure4",
        "text_dir": "PATH/TO/OUTPUT",
        "tmp_dir": "PATH/TO/OUTPUT/_tmp",
    },
    "duckdb": {
        "memory_limit": "32GB",
        "threads": 8,
    },
    "sample_denom": 500,
    "dis_min": -55,
    "dis_max": 55,
}

PARQUET_A = CONFIG["input"]["parquet_a"]
PARQUET_T = CONFIG["input"]["parquet_t"]
REF_GENOME = CONFIG["input"]["ref_genome"]

FIG_DIR = CONFIG["output"]["fig_dir"]
TEXT_DIR = CONFIG["output"]["text_dir"]
TMP_DIR = CONFIG["output"]["tmp_dir"]

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

FIG_4A_ABSDIS = os.path.join(FIG_DIR, "figure4A_absdis_vs_median.png")
FIG_4B_COMBINED = os.path.join(FIG_DIR, "figure4B_count_ratio_combined.png")
FIG_4C = os.path.join(FIG_DIR, "Figure4C_dis+1.png")
FIG_4C_HEATMAP = os.path.join(FIG_DIR, "Figure4C_heatmap_dis+1.png")
FIG_4D = os.path.join(FIG_DIR, "Figure4D_dis-1.png")
FIG_4E = os.path.join(FIG_DIR, "Figure4E_boxplot_GroupA_vs_0.png")
FIG_4F = os.path.join(FIG_DIR, "Figure4F_top15_triplet_transitions.png")

STATS_C = os.path.join(TEXT_DIR, "C_ref_base_stats_dis+1.tsv")
STATS_D = os.path.join(TEXT_DIR, "D_ref_base_stats_dis-1.tsv")
E_DIS1_SUMMARY = os.path.join(TEXT_DIR, "E_dis1_summary_stats.csv")
E_DIS1_DUNN = os.path.join(TEXT_DIR, "E_dis1_dunn_holm.csv")
E_TRIPLET_SUMMARY = os.path.join(TEXT_DIR, "E_triplet_transition_summary.tsv")
E_PAIRWISE_TESTS = os.path.join(TEXT_DIR, "E_transition_pairwise_tests.tsv")
CENTER_MISMATCH_CSV = os.path.join(TEXT_DIR, "E_center_mismatch_samples.csv")
REF_MISMATCH_CSV = os.path.join(TEXT_DIR, "E_ref_mismatch_samples.csv")

DIS_MIN = CONFIG["dis_min"]
DIS_MAX = CONFIG["dis_max"]
SAMPLE_DENOM = CONFIG["sample_denom"]

A_YLIM = None
A_YLIM_PAD = 0.08
FIG_A_SIZE = (9, 6)
FIG_A2_SIZE = (12, 6)
FIG_B2_SIZE = (14, 6)
DOT_SIZE = 60
LABEL_FS = 11
TITLE_FS = 12
TICK_FS = 10
LEGEND_FS = 10
ANNOT_FS = 9
MAIN_COLOR = "#1f77b4"
RATIO_LS = ["--", "-"]

CUSTOM_ORDER_C = ["A", "T", "G", "C"]
CUSTOM_ORDER_D = ["A/T/C", "G"]
BLUE_DARK = "#1f77b4"
BLUE_LIGHT = "#9ecae1"
BUBBLE_RANGE = (-0.005, 0.002)

MUT_SET_DIS1 = {
    "G>C": "GC", "C>G": "CG", "G>A": "GA", "G>T": "GT",
    "C>A": "CA", "C>T": "CT", "A>T": "AT", "T>A": "TA",
}

COMPLEMENT_MAP = {
    "A": "T", "T": "A", "G": "C", "C": "G",
    "a": "t", "t": "a", "g": "c", "c": "g",
    "N": "N", "n": "n",
}

GROUP_COLORS = {"GroupA": BLUE_DARK, "GroupB": BLUE_LIGHT}
GROUP_ORDER = ["GroupA", "GroupB"]
CONVERT_TO_RNA = True
DIFF_THRESHOLD = None


def init_duckdb():
    con = duckdb.connect()
    con.execute(f"SET threads = {CONFIG['duckdb']['threads']};")
    con.execute(f"SET memory_limit = '{CONFIG['duckdb']['memory_limit']}';")
    con.execute(f"SET temp_directory = '{TMP_DIR}';")
    con.execute("SET preserve_insertion_order = false;")
    return con


def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def dynamic_ylim(vals, pad_ratio=A_YLIM_PAD, quantile_range=None, clip_lower=None):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    if quantile_range is not None and len(v) >= 10:
        lo, hi = np.percentile(v, list(quantile_range))
    else:
        lo, hi = float(v.min()), float(v.max())
    if clip_lower is not None:
        lo = max(lo, clip_lower)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi) * 0.1, 1e-3)
    pad = span * pad_ratio
    return lo - pad, hi + pad


def safe_polyfit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) <= 1e-9:
        return None
    try:
        return np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return None


def jonckheere_terpstra_test(groups, alternative="decreasing"):
    groups = [np.asarray(g, dtype=float) for g in groups]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan, np.nan
    k = len(groups)
    ns = [len(g) for g in groups]
    jt_stat = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            u, _ = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            jt_stat += u
    N = sum(ns)
    mu = (N ** 2 - sum(n ** 2 for n in ns)) / 4
    term1 = N ** 2 * (2 * N + 3)
    term2 = sum(n ** 2 * (2 * n + 3) for n in ns)
    sigma2 = (term1 - term2) / 72
    sigma = np.sqrt(sigma2) if sigma2 > 0 else np.nan
    z = (jt_stat - mu) / sigma if (sigma and not np.isnan(sigma)) else np.nan
    if alternative == "decreasing":
        p = norm.sf(z) if not np.isnan(z) else np.nan
    else:
        p = norm.cdf(z) if not np.isnan(z) else np.nan
    jt_max = sum(ns[i] * ns[j] for i in range(k) for j in range(i + 1, k))
    effect = (jt_stat - jt_max / 2) / (jt_max / 2) if jt_max > 0 else np.nan
    return jt_stat, p, effect, jt_max


def format_count(x):
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.0f}k"
    else:
        return str(int(x))


def _pick_digits(span):
    if span < 0.0005:
        return 5
    if span < 0.005:
        return 4
    if span < 0.02:
        return 4
    if span < 0.5:
        return 3
    return 2


def configure_colorbar(cbar, norm):
    vmin, vcenter, vmax = norm.vmin, norm.vcenter, norm.vmax
    span_neg = abs(vmin - vcenter)
    span_pos = abs(vmax - vcenter)
    span = max(span_neg, span_pos) * 2 if max(span_neg, span_pos) > 0 else 1e-9
    digits = _pick_digits(span)
    ratio = (max(span_neg, span_pos) / min(span_neg, span_pos)
             if span_neg > 0 and span_pos > 0 else 1.0)
    ticks = [vmin, vcenter, vmax] if ratio > 5 else [
        vmin, (vmin + vcenter) / 2, vcenter, (vcenter + vmax) / 2, vmax
    ]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.{digits}f}"))
    cbar.ax.tick_params(labelsize=8)


def _boxplot_stats(vals):
    if len(vals) < 2:
        return None
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    in_lo = vals[vals >= q1 - 1.5 * iqr]
    in_hi = vals[vals <= q3 + 1.5 * iqr]
    return {
        "med": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "whislo": float(in_lo.min()) if len(in_lo) else float(q1),
        "whishi": float(in_hi.max()) if len(in_hi) else float(q3),
        "fliers": [],
    }


def load_figure4ab_data():
    print("Loading data for Figure4A/4B ...")
    for p in (PARQUET_A, PARQUET_T):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input: {p}")

    con = init_duckdb()

    print("Computing pos_full ...")
    pos_sql = f"""
    WITH u AS (
        SELECT dis, diff FROM read_parquet('{PARQUET_A}')
         WHERE diff IS NOT NULL AND dis IS NOT NULL AND dis != 0
           AND dis BETWEEN {DIS_MIN} AND {DIS_MAX}
        UNION ALL
        SELECT dis, diff FROM read_parquet('{PARQUET_T}')
         WHERE diff IS NOT NULL AND dis IS NOT NULL AND dis != 0
           AND dis BETWEEN {DIS_MIN} AND {DIS_MAX}
    )
    SELECT dis AS distance,
           COUNT(*) AS n,
           approx_quantile(ABS(diff), 0.25) AS q25_abs,
           approx_quantile(ABS(diff), 0.50) AS median_abs,
           approx_quantile(ABS(diff), 0.75) AS q75_abs,
           AVG(CASE WHEN ABS(diff) > 0.05 THEN 1.0 ELSE 0.0 END) AS r_abs_005,
           AVG(CASE WHEN ABS(diff) > 0.10 THEN 1.0 ELSE 0.0 END) AS r_abs_010,
           SUM(CASE WHEN ABS(diff) > 0.10 THEN 1 ELSE 0 END) AS cnt_gt010
    FROM u
    GROUP BY dis
    ORDER BY dis
    """
    pos_full = con.execute(pos_sql).df()
    all_dis = pd.DataFrame({"distance": [d for d in range(DIS_MIN, DIS_MAX + 1) if d != 0]})
    pos_full = all_dis.merge(pos_full, on="distance", how="left")
    zero = pos_full["n"].fillna(0) == 0
    for c in ["q25_abs", "median_abs", "q75_abs", "r_abs_005", "r_abs_010"]:
        pos_full.loc[zero, c] = np.nan
    for c in ["n", "cnt_gt010"]:
        pos_full[c] = pos_full[c].fillna(0).astype(int)
    pos_full["direction"] = np.where(pos_full["distance"] < 0, "upstream", "downstream")
    pos_full["abs_distance"] = pos_full["distance"].abs()
    print(f"pos_full: {len(pos_full)} positions, {int(pos_full['n'].sum()):,} SNP")

    print("Computing pooled_full ...")
    pooled_sql = f"""
    WITH u AS (
        SELECT dis, diff FROM read_parquet('{PARQUET_A}')
         WHERE diff IS NOT NULL AND dis IS NOT NULL AND dis != 0
           AND dis BETWEEN {DIS_MIN} AND {DIS_MAX}
        UNION ALL
        SELECT dis, diff FROM read_parquet('{PARQUET_T}')
         WHERE diff IS NOT NULL AND dis IS NOT NULL AND dis != 0
           AND dis BETWEEN {DIS_MIN} AND {DIS_MAX}
    )
    SELECT ABS(dis) AS abs_distance,
           COUNT(*) AS n,
           approx_quantile(ABS(diff), 0.50) AS median_abs
    FROM u
    GROUP BY ABS(dis)
    ORDER BY abs_distance
    """
    pooled_full = con.execute(pooled_sql).df()
    all_abs = pd.DataFrame({"abs_distance": range(1, DIS_MAX + 1)})
    pooled_full = all_abs.merge(pooled_full, on="abs_distance", how="left")
    zero_a = pooled_full["n"].fillna(0) == 0
    pooled_full.loc[zero_a, "median_abs"] = np.nan
    pooled_full["n"] = pooled_full["n"].fillna(0).astype(int)
    print(f"pooled_full: {len(pooled_full)} positions, {int(pooled_full['n'].sum()):,} SNP")

    print(f"Sampling (hash % {SAMPLE_DENOM} == 0) ...")
    sample_sql = f"""
    WITH u AS (
        SELECT chrom, "start" AS start, "end" AS end, dis, diff
        FROM read_parquet('{PARQUET_A}')
        WHERE diff IS NOT NULL AND dis IS NOT NULL AND dis != 0
          AND dis BETWEEN {DIS_MIN} AND {DIS_MAX}
        UNION ALL
        SELECT chrom, "start" AS start, "end" AS end, dis, diff
        FROM read_parquet('{PARQUET_T}')
        WHERE diff IS NOT NULL AND dis IS NOT NULL AND dis != 0
          AND dis BETWEEN {DIS_MIN} AND {DIS_MAX}
    )
    SELECT dis AS distance, diff AS delta
    FROM u
    WHERE (hash(chrom, "start", "end", dis) % {SAMPLE_DENOM}) = 0
    """
    sampled = con.execute(sample_sql).df()
    con.close()

    if len(sampled) > 0:
        sampled["abs_distance"] = sampled["distance"].abs()
        sampled["delta_abs"] = sampled["delta"].abs()
        sampled["direction"] = np.where(sampled["distance"] < 0, "upstream", "downstream")

    print(f"sampled: {len(sampled):,} rows")
    return sampled, pos_full, pooled_full


def step_A_abs_dis_trend(sampled, pooled_full, prefix):
    print("STEP A: overall trend (|dis|)")
    if len(sampled) < 2:
        print("SKIP (sampled empty)")
        return {}

    sorted_abs = sorted(sampled["abs_distance"].unique())
    groups = [
        sampled[sampled["abs_distance"] == d]["delta_abs"].dropna().values
        for d in sorted_abs
    ]
    groups = [g for g in groups if len(g) >= 2]

    print(f"J-T (|dis|): positions={len(groups)}, sampled records={len(sampled):,}")
    jt_stat, jt_p, jt_eff, jt_max = jonckheere_terpstra_test(groups, alternative="decreasing")
    print(f"J-T(|delta|): stat={jt_stat:.4f}, max={jt_max}, effect={jt_eff:.4f}, p={jt_p:.4e}")

    pf_valid = pooled_full.dropna(subset=["median_abs"])
    if len(pf_valid) >= 2:
        rho_pos, p_pos = spearmanr(pf_valid["abs_distance"], pf_valid["median_abs"])
        print(f"Spearman per-position: rho={rho_pos:.4f}, p={p_pos:.4e}, n_pos={len(pf_valid)}")
    else:
        rho_pos = p_pos = np.nan

    rho_rec, p_rec = spearmanr(sampled["abs_distance"], sampled["delta_abs"])
    print(f"Spearman per-record (sampled): rho={rho_rec:.4f}, p={p_rec:.4e}")

    result = {
        "scope": "abs_dis_full_pooled",
        "jt_stat": jt_stat,
        "jt_p": jt_p,
        "jt_effect": jt_eff,
        "jt_max": jt_max,
        "spearman_rho_per_record": rho_rec,
        "spearman_p_per_record": p_rec,
        "spearman_rho_per_position": rho_pos,
        "spearman_p_per_position": p_pos,
        "n_positions": len(groups),
        "n_records_sampled": len(sampled),
    }
    pd.DataFrame([result]).to_csv(
        f"{prefix}_stepA_abs_dis_trend.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    print(f"Saved: {prefix}_stepA_abs_dis_trend.tsv")
    return result


def plot_A_absdis_vs_median(pooled_full, stepA_result):
    print("Plot figure4A ...")
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=FIG_A_SIZE, dpi=300)

    data = pooled_full.dropna(subset=["median_abs"]).copy()
    if len(data) == 0:
        print("SKIP (empty)")
        plt.close(fig)
        return

    x = data["abs_distance"].values.astype(float)
    y = data["median_abs"].values.astype(float)

    ax.scatter(x, y, s=DOT_SIZE, color=MAIN_COLOR, alpha=0.78,
               edgecolor="white", linewidth=0.5, zorder=3)

    z = safe_polyfit(x, y)
    if z is not None:
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, np.poly1d(z)(xfit), color=MAIN_COLOR,
                lw=1.6, alpha=0.55, ls="--", zorder=2)

    ylim = A_YLIM if A_YLIM is not None else dynamic_ylim(y, pad_ratio=A_YLIM_PAD, clip_lower=0)
    if ylim:
        ax.set_ylim(*ylim)

    if stepA_result:
        anno = (
            f"J-T (|dis|): effect={stepA_result.get('jt_effect', np.nan):.3f}, "
            f"p={stepA_result.get('jt_p', np.nan):.2e}\n"
            f"Spearman rho(|dis|, median|delta|)="
            f"{stepA_result.get('spearman_rho_per_position', np.nan):.3f}, "
            f"p={stepA_result.get('spearman_p_per_position', np.nan):.2e}"
        )
        ax.text(0.98, 0.96, anno, transform=ax.transAxes, fontsize=ANNOT_FS,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                          edgecolor=MAIN_COLOR, linewidth=0.8))

    ax.set_title("figure4A. Median |delta score| vs |distance|",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("|Distance| to editing site (bp)", fontsize=LABEL_FS)
    ax.set_ylabel("Median |delta score|", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xlim(0, 56)

    fig.tight_layout()
    save_fig(fig, FIG_4A_ABSDIS)


def plot_B_count_ratio_combined(pos_full):
    print("Plot figure4B ...")
    plt.style.use("seaborn-v0_8-white")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIG_B2_SIZE, dpi=300)

    x = pos_full["distance"].values
    n = pos_full["n"].values

    v_005 = pd.Series(pos_full["r_abs_005"] * 100).where(n > 0).values
    v_010 = pd.Series(pos_full["r_abs_010"] * 100).where(n > 0).values
    ax_left.plot(x, v_005, color=MAIN_COLOR, linestyle=RATIO_LS[0], linewidth=2,
                 label="|delta| > 0.05")
    ax_left.plot(x, v_010, color=MAIN_COLOR, linestyle=RATIO_LS[1], linewidth=2,
                 label="|delta| > 0.10")
    ax_left.set_xlabel("Distance to editing site (bp)", fontsize=LABEL_FS)
    ax_left.set_ylabel("Proportion (%)", fontsize=LABEL_FS)
    ax_left.set_title("B1. Proportion with |delta| above threshold",
                      fontsize=TITLE_FS, fontweight="bold")
    ax_left.set_xlim(DIS_MIN, DIS_MAX)
    ax_left.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax_left.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_left.legend(loc="best", fontsize=LEGEND_FS)
    ax_left.tick_params(axis="both", labelsize=TICK_FS)

    cnt = pos_full["cnt_gt010"].values
    mask = cnt > 0
    ax_right.bar(x, np.where(mask, cnt, 0), width=0.8,
                 color=MAIN_COLOR, alpha=0.7)
    ax_right.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax_right.set_xlabel("Distance to editing site (bp)", fontsize=LABEL_FS)
    ax_right.set_ylabel("Number of SNPs", fontsize=LABEL_FS)
    ax_right.set_title("B2. Count of SNPs with |delta| > 0.1",
                       fontsize=TITLE_FS, fontweight="bold")
    ax_right.set_xlim(DIS_MIN, DIS_MAX)
    ax_right.tick_params(axis="both", labelsize=TICK_FS)

    fig.tight_layout()
    save_fig(fig, FIG_4B_COMBINED)


def load_cd_stats():
    print("Loading Figure4C/4D stats ...")
    for p in (PARQUET_A, PARQUET_T):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input: {p}")

    con = init_duckdb()
    sql = f"""
    WITH u AS (
        SELECT dis, strand, SUBSTR(snptype, 1, 1) AS raw_ref, diff
        FROM read_parquet('{PARQUET_A}')
        WHERE dis IN (-1, 1) AND diff IS NOT NULL
        UNION ALL
        SELECT dis, strand, SUBSTR(snptype, 1, 1) AS raw_ref, diff
        FROM read_parquet('{PARQUET_T}')
        WHERE dis IN (-1, 1) AND diff IS NOT NULL
    )
    SELECT
        dis,
        CASE
            WHEN strand = '+' THEN raw_ref
            ELSE CASE raw_ref
                WHEN 'A' THEN 'T' WHEN 'T' THEN 'A'
                WHEN 'G' THEN 'C' WHEN 'C' THEN 'G'
                ELSE raw_ref END
        END AS ref_base,
        COUNT(*)                                  AS count,
        MEDIAN(diff)                              AS median_diff,
        AVG(diff)                                 AS avg_diff,
        SUM(CASE WHEN diff < 0 THEN 1 ELSE 0 END) AS n_loss,
        SUM(CASE WHEN diff > 0 THEN 1 ELSE 0 END) AS n_gain
    FROM u
    WHERE raw_ref IN ('A','T','G','C')
    GROUP BY dis, ref_base
    ORDER BY dis, ref_base
    """
    df = con.execute(sql).df()
    con.close()
    return df


def plot_panel(df_plot, out_path, fig_label, dis_label_text):
    print(f"Plotting {os.path.basename(out_path)} ...")
    order = df_plot["ref_base"].tolist()

    fig, (ax_b, ax_p) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.32})

    vmin, vmax = BUBBLE_RANGE
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    counts = df_plot["count"].values.astype(float)
    diff_vals = df_plot["median_diff"].values.astype(float)

    counts_valid = counts[counts > 0]
    if len(counts_valid) == 0:
        counts_valid = np.array([1.0])
    min_size, max_size = 400, 2000
    min_c, max_c = np.sqrt(counts_valid.min()), np.sqrt(counts_valid.max())

    def bubble_size(x):
        if max_c == min_c:
            return (min_size + max_size) / 2
        return min_size + (np.sqrt(x) - min_c) / (max_c - min_c) * (max_size - min_size)

    for j, base in enumerate(order):
        row = df_plot.iloc[j]
        cnt, dif = row["count"], row["median_diff"]
        if cnt > 0 and not np.isnan(dif):
            ax_b.scatter(j, 0, s=bubble_size(cnt),
                         c=[dif], cmap=cmap, norm=norm,
                         edgecolor="black", linewidth=0.8)
        else:
            ax_b.scatter(j, 0, s=150, color="lightgrey",
                         edgecolor="grey", linewidth=0.4)
        if cnt > 0:
            ax_b.text(j, -0.55, format_count(int(cnt)),
                      ha="center", va="top", fontsize=10,
                      fontweight="bold", color="#333")

    ax_b.set_xlim(-0.5, len(order) - 0.5)
    ax_b.set_ylim(-0.85, 0.55)
    ax_b.set_xticks(range(len(order)))
    ax_b.set_xticklabels(order, fontsize=13, fontweight="bold")
    ax_b.set_yticks([])
    ax_b.set_xlabel("Ref base (editing-direction)", fontsize=11)
    ax_b.set_title("Bubble: median delta S at ref base\n(size = count)",
                   fontsize=11, fontweight="bold")
    ax_b.grid(False)
    for sp in ["top", "right", "left"]:
        ax_b.spines[sp].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_b, fraction=0.045, pad=0.04)
    cbar.set_label("median delta S", fontsize=9)
    configure_colorbar(cbar, norm)

    n_total_arr = df_plot["count"].values.astype(int)
    n_loss_arr = df_plot["n_loss"].values.astype(int)
    n_gain_arr = df_plot["n_gain"].values.astype(int)
    p_loss = np.where(n_total_arr > 0, n_loss_arr / n_total_arr * 100, 0.0)
    p_gain = np.where(n_total_arr > 0, n_gain_arr / n_total_arr * 100, 0.0)

    x = np.arange(len(order))
    w = 0.38
    ax_p.bar(x - w / 2, p_loss, w, color=BLUE_DARK, edgecolor="none",
             label="Loss (delta S < 0)")
    ax_p.bar(x + w / 2, p_gain, w, color=BLUE_LIGHT, edgecolor="none",
             label="Gain (delta S > 0)")

    y_top = max(float(np.max(p_loss)), float(np.max(p_gain)), 1.0) * 1.18
    ax_p.set_ylim(0, y_top)

    THRESH = 10.0
    for i in range(len(order)):
        if n_loss_arr[i] > 0:
            lbl = format_count(int(n_loss_arr[i]))
            if p_loss[i] >= THRESH:
                ax_p.text(x[i] - w / 2, p_loss[i] - 1.2, lbl, ha="center", va="top",
                          fontsize=8, color="white", fontweight="bold", zorder=5)
            else:
                ax_p.text(x[i] - w / 2, p_loss[i] + 1.2, lbl, ha="center", va="bottom",
                          fontsize=8, color=BLUE_DARK, fontweight="bold", zorder=5)
        if n_gain_arr[i] > 0:
            lbl = format_count(int(n_gain_arr[i]))
            if p_gain[i] >= THRESH:
                ax_p.text(x[i] + w / 2, p_gain[i] - 1.2, lbl, ha="center", va="top",
                          fontsize=8, color=BLUE_DARK, fontweight="bold", zorder=5)
            else:
                ax_p.text(x[i] + w / 2, p_gain[i] + 1.2, lbl, ha="center", va="bottom",
                          fontsize=8, color=BLUE_DARK, fontweight="bold", zorder=5)

    ax_p.set_xticks(x)
    ax_p.set_xticklabels(order, fontsize=13, fontweight="bold")
    ax_p.set_xlabel("Ref base (editing-direction)", fontsize=11)
    ax_p.set_ylabel("Proportion within ref base (%)", fontsize=11)
    ax_p.set_title("Loss / Gain proportions by ref base",
                   fontsize=11, fontweight="bold")
    ax_p.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.9)
    ax_p.grid(False)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)
    ax_p.tick_params(axis="both", labelsize=10)

    fig.suptitle(f"{fig_label}. {dis_label_text}, n_total = {int(n_total_arr.sum()):,}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_figure4c_4d(df_raw_stats):
    df_plus = df_raw_stats[df_raw_stats["dis"] == 1].copy()
    df_minus = df_raw_stats[df_raw_stats["dis"] == -1].copy()

    agg_C = pd.DataFrame({"ref_base": CUSTOM_ORDER_C}).merge(
        df_plus, on="ref_base", how="left")
    agg_C[["count", "n_loss", "n_gain"]] = (
        agg_C[["count", "n_loss", "n_gain"]].fillna(0).astype(int)
    )
    agg_C.to_csv(STATS_C, sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {STATS_C}")

    map_D = {"A": "A/T/C", "T": "A/T/C", "C": "A/T/C", "G": "G"}
    df_minus["group"] = df_minus["ref_base"].map(map_D)
    agg_D = df_minus.groupby("group").agg(
        count=("count", "sum"),
        median_diff=("median_diff", "median"),
        avg_diff=("avg_diff", "mean"),
        n_loss=("n_loss", "sum"),
        n_gain=("n_gain", "sum"),
    ).reset_index().rename(columns={"group": "ref_base"})
    agg_D = pd.DataFrame({"ref_base": CUSTOM_ORDER_D}).merge(
        agg_D, on="ref_base", how="left")
    agg_D[["count", "n_loss", "n_gain"]] = (
        agg_D[["count", "n_loss", "n_gain"]].fillna(0).astype(int)
    )
    agg_D.to_csv(STATS_D, sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {STATS_D}")

    plot_panel(agg_C, FIG_4C, "Figure4C", "dis=+1 (downstream)")
    plot_panel(agg_D, FIG_4D, "Figure4D", "dis=-1 (upstream)")


def load_heatmap_data(con, parquet_paths, pred_source, dis, mut_set, label):
    snp_types = list(mut_set.values())
    in_clause = ",".join(f"'{s}'" for s in snp_types)

    parts = []
    for path in parquet_paths:
        parts.append(f"""
        SELECT
            substr(snptype, 1, 1) || '>' || substr(snptype, 2, 1) AS mutation,
            diff AS diff
        FROM read_parquet('{path}')
        WHERE dis = {dis}
          AND pred_source = '{pred_source}'
          AND snptype IN ({in_clause})
          AND diff IS NOT NULL
        """)
    sql = " UNION ALL ".join(parts)
    df = con.execute(sql).df()
    df = df[df["mutation"].isin(mut_set.keys())]
    df["dataset"] = label
    return df


def compute_summary(df, all_muts):
    summary = []
    for mut in all_muts:
        sub = df[df["mutation"] == mut]["diff"]
        if len(sub) == 0:
            summary.append([mut, 0, np.nan, np.nan])
        else:
            q1, q3 = np.percentile(sub, [25, 75])
            summary.append([mut, len(sub), np.median(sub), q3 - q1])
    return pd.DataFrame(summary, columns=["mutation", "n", "median", "IQR"])


def dunn_test_holm_scipy(df, all_muts, val_col="diff", group_col="mutation"):
    if not USE_SP:
        raise ImportError("Please install scikit-posthocs.")

    present = [m for m in all_muts if (df[group_col] == m).any()]
    if len(present) < 2:
        raise ValueError(f"Valid groups < 2: {present}")

    sub = df[df[group_col].isin(present)]
    pvals = sp.posthoc_dunn(sub, val_col=val_col, group_col=group_col, p_adjust="holm")
    dunn_rows = []
    groups = pvals.columns.tolist()
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            p = pvals.loc[g1, g2]
            sig = "ns"
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            dunn_rows.append([g1, g2, np.nan, np.nan, p, sig])
    dunn_df = pd.DataFrame(
        dunn_rows,
        columns=["pair1", "pair2", "stat_z", "raw_p", "holm_adj_p", "significance"],
    )

    groups_data = [sub[sub[group_col] == g][val_col].values for g in groups]
    kw_stat, kw_p = kruskal(*groups_data)
    return kw_stat, kw_p, dunn_df


def plot_heatmap(summary_df, all_muts, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    medians = summary_df.set_index("mutation")["median"]
    n_muts = len(all_muts)
    diff_arr = np.zeros((n_muts, n_muts))
    for i, m1 in enumerate(all_muts):
        for j, m2 in enumerate(all_muts):
            v1 = medians.get(m1, np.nan)
            v2 = medians.get(m2, np.nan)
            diff_arr[i, j] = v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan
    diff_mat = pd.DataFrame(diff_arr, index=all_muts, columns=all_muts)

    vals = diff_mat.values.flatten()
    vals = vals[~np.isnan(vals)]
    max_abs = max(abs(vals.min()), abs(vals.max())) if len(vals) > 0 else 0.001
    if max_abs == 0:
        max_abs = 0.001

    mask = np.triu(np.ones_like(diff_mat, dtype=bool), k=0)

    sns.heatmap(
        diff_mat, annot=False,
        cmap="RdBu",
        mask=mask, linewidths=0.5,
        vmin=-max_abs, vmax=max_abs,
        cbar_kws={
            "label": "Signed delta median(diff)  (row - column)\n"
                     "red: < 0 | blue: > 0",
            "shrink": 0.8,
        },
        ax=ax,
    )
    ax.set_title("Pairwise delta median(diff) - dis = +1\n"
                 "(red: < 0; blue: > 0)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    save_fig(fig, out_path)


def process_heatmap_dataset(df, all_muts, label, out_fig_path, text_prefix):
    print(f"Processing heatmap {label} (N={len(df):,})")
    n_per_mut = df["mutation"].value_counts()
    present = [m for m in all_muts if n_per_mut.get(m, 0) >= 2]
    if len(present) < 2:
        print(f"SKIP valid groups < 2: {dict(n_per_mut)}")
        return
    df_filt = df[df["mutation"].isin(present)].copy()
    print(f"Kept groups: {present}")
    print(f"Sample size per group: {n_per_mut[present].to_dict()}")

    summary = compute_summary(df_filt, all_muts)
    kw_stat, kw_p, dunn = dunn_test_holm_scipy(df_filt, all_muts)
    print(f"Kruskal-Wallis (on diff): H = {kw_stat:.2f}, p = {kw_p:.4e}")

    summary.to_csv(os.path.join(TEXT_DIR, f"{text_prefix}_summary_stats.csv"), index=False)
    dunn.to_csv(os.path.join(TEXT_DIR, f"{text_prefix}_dunn_holm.csv"), index=False)
    print(f"Saved: {os.path.join(TEXT_DIR, text_prefix + '_summary_stats.csv')}")
    print(f"Saved: {os.path.join(TEXT_DIR, text_prefix + '_dunn_holm.csv')}")

    plot_heatmap(summary, present, out_fig_path)


def plot_figure4c_heatmap():
    print("Loading Figure4C heatmap data ...")
    con = init_duckdb()
    mut_set1 = MUT_SET_DIS1
    all_muts1 = list(mut_set1.keys())

    df_snp_1 = load_heatmap_data(
        con, [PARQUET_A, PARQUET_T], "snp", dis=1,
        mut_set=mut_set1, label="snp_edit")
    df_edit_1 = load_heatmap_data(
        con, [PARQUET_A, PARQUET_T], "edit", dis=1,
        mut_set=mut_set1, label="edit_edit")
    df_all_1 = pd.concat([df_snp_1, df_edit_1], ignore_index=True)
    print(f"dis=+1: snp={len(df_snp_1):,}, edit={len(df_edit_1):,}, merged={len(df_all_1):,}")

    process_heatmap_dataset(
        df_all_1, all_muts1, "dis=+1",
        out_fig_path=FIG_4C_HEATMAP,
        text_prefix="E_dis1",
    )
    con.close()


def load_di1_meta():
    print("Step 1a: extract dis=+/-1 data ...")
    con = init_duckdb()
    sql = f"""
    WITH u AS (
        SELECT chrom, "start" AS start, "end" AS end,
               snptype, dis, diff, strand, modifiedscore, editscore
        FROM read_parquet('{PARQUET_A}')
        WHERE dis IN (-1, 1) AND editscore IS NOT NULL
          AND modifiedscore IS NOT NULL AND diff IS NOT NULL
        UNION ALL
        SELECT chrom, "start" AS start, "end" AS end,
               snptype, dis, diff, strand, modifiedscore, editscore
        FROM read_parquet('{PARQUET_T}')
        WHERE dis IN (-1, 1) AND editscore IS NOT NULL
          AND modifiedscore IS NOT NULL AND diff IS NOT NULL
    )
    SELECT
        chrom, start, "end", snptype, dis, diff, strand,
        modifiedscore, editscore,
        CASE WHEN strand = '+' THEN start + 55 ELSE "end" - 56 END AS edit_pos,
        CASE WHEN strand = '+' THEN start + 54 ELSE "end" - 57 END AS trip_start,
        CASE WHEN strand = '+' THEN start + 57 ELSE "end" - 54 END AS trip_end
    FROM u
    """
    df = con.execute(sql).df()
    con.close()
    print(f"dis=+/-1 rows: {len(df):,}")
    return df


def extract_original_triplets(df):
    print("Step 1b: bedtools extract original triplets ...")
    df = df.copy().reset_index(drop=True)
    unique_regions = (
        df[["chrom", "trip_start", "trip_end", "strand"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Before dedup: {len(df):,} -> after dedup: {len(unique_regions):,}")

    bed_lines = [
        f"{r['chrom']}\t{r['trip_start']}\t{r['trip_end']}\t.\t0\t{r['strand']}"
        for _, r in unique_regions.iterrows()
    ]
    bed_text = "\n".join(bed_lines) + "\n"
    cmd = ["bedtools", "getfasta", "-fi", REF_GENOME, "-bed", "stdin", "-tab", "-s"]
    result = subprocess.run(cmd, input=bed_text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"bedtools error:\n{result.stderr}")

    seqs = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        header, seq = parts[0], parts[1].upper()
        try:
            if header.endswith("(+)") or header.endswith("(-)"):
                strand = header[-2]
                coord = header[:-3]
            else:
                strand = "+"
                coord = header
            chrom, rest = coord.split(":", 1)
            s_str, e_str = rest.split("-", 1)
            seqs[(chrom, int(s_str), int(e_str), strand)] = seq
        except Exception:
            continue

    print(f"bedtools returned {len(seqs):,} unique sequences")
    df["original_triplet"] = df.apply(
        lambda r: seqs.get((r["chrom"], int(r["trip_start"]),
                            int(r["trip_end"]), r["strand"])), axis=1)
    df = df.dropna(subset=["original_triplet"]).copy()
    df = df[df["original_triplet"].str.len() == 3].copy()
    print(f"Matched and length=3: {len(df):,}")
    return df


def make_modified_triplet(original_trip, dis, snptype, strand):
    if not isinstance(original_trip, str) or len(original_trip) != 3:
        return None
    if not isinstance(snptype, str) or len(snptype) < 2:
        return None
    alt_base = snptype[1]
    if strand == "-":
        alt_base = COMPLEMENT_MAP.get(alt_base, alt_base)
    if dis == 1:
        return original_trip[:2] + alt_base
    elif dis == -1:
        return alt_base + original_trip[1:]
    return None


def build_triplets():
    df_meta = load_di1_meta()
    df_meta = extract_original_triplets(df_meta)
    print("Step 1c: generate modified_triplet ...")
    df = df_meta.copy()
    df["modified_triplet"] = df.apply(
        lambda r: make_modified_triplet(r["original_triplet"], r["dis"],
                                        r["snptype"], r["strand"]), axis=1)
    df = df.dropna(subset=["modified_triplet"])
    df = df[df["modified_triplet"].str.len() == 3].copy()
    return df


def validate_triplets(df):
    print("Step 2: double validation ...")
    df = df.copy()
    n_total = len(df)

    df["center_base"] = df["original_triplet"].str[1]
    df["center_is_A"] = df["center_base"] == "A"
    n_center_bad = int((~df["center_is_A"]).sum())
    print(f"Check 1 (editing site = A): failed {n_center_bad:,} / {n_total:,}")

    df["snp_offset"] = np.where(df["dis"] == 1, 2, 0)

    def expected_ref(snptype, strand):
        ref = snptype[0]
        return COMPLEMENT_MAP.get(ref, ref) if strand == "-" else ref

    df["expected_ref"] = df.apply(
        lambda r: expected_ref(r["snptype"], r["strand"]), axis=1)
    df["actual_ref"] = df.apply(
        lambda r: r["original_triplet"][int(r["snp_offset"])], axis=1)
    df["ref_match"] = df["expected_ref"] == df["actual_ref"]
    n_ref_bad = int((~df["ref_match"]).sum())
    print(f"Check 2 (SNP site ref): failed {n_ref_bad:,} / {n_total:,}")

    if n_center_bad > 0:
        df[~df["center_is_A"]].head(50).to_csv(CENTER_MISMATCH_CSV, index=False)
    if n_ref_bad > 0:
        df[~df["ref_match"]].head(50).to_csv(REF_MISMATCH_CSV, index=False)

    df_clean = df[df["center_is_A"] & df["ref_match"]].copy()
    df_clean = df_clean.drop(columns=[
        "center_base", "center_is_A", "snp_offset",
        "expected_ref", "actual_ref", "ref_match",
    ])
    print(f"Both checks passed: {len(df_clean):,} rows "
          f"({100 * len(df_clean) / max(1, n_total):.2f}%)")
    return df_clean


def preprocess(df, diff_threshold=DIFF_THRESHOLD):
    print(f"Step 3: preprocessing (|delta S| filter={diff_threshold}) ...")
    df = df.copy()
    n_before = len(df)

    df["delta_score"] = df["modifiedscore"] - df["editscore"]
    diff_check = (df["delta_score"] - df["diff"]).abs()
    n_bad_diff = int((diff_check > 1e-6).sum())
    if n_bad_diff > 0:
        print(f"WARN: {n_bad_diff:,} rows delta_score != diff")
    else:
        print("delta S = modifiedscore - editscore (= diff)")

    if diff_threshold is not None and diff_threshold > 0:
        mask = df["diff"].abs() >= diff_threshold
        df = df[mask].copy()
        print(f"|diff| >= {diff_threshold}: {n_before:,} -> {len(df):,}")

    if len(df) == 0:
        return None

    df["center_base"] = df["modified_triplet"].str[1]
    df = df[df["center_base"] == "A"].copy()
    print(f"ADAR sites: {len(df):,}")

    if CONVERT_TO_RNA:
        df["original_triplet"] = df["original_triplet"].str.replace("T", "U")
        df["modified_triplet"] = df["modified_triplet"].str.replace("T", "U")

    def assign_group(trip):
        if not isinstance(trip, str) or len(trip) != 3:
            return None
        b5 = trip[0]
        b3 = trip[2]
        if b5 != "G" and b3 in {"G", "C"}:
            return "GroupA"
        else:
            return "GroupB"

    df["trip_group"] = df["original_triplet"].apply(assign_group)
    df = df.dropna(subset=["trip_group"])
    df["trip_group"] = pd.Categorical(
        df["trip_group"], categories=GROUP_ORDER, ordered=True)

    df["is_disrupted"] = df["original_triplet"] != df["modified_triplet"]
    df["snp_subst"] = df.apply(
        lambda r: f"{r['original_triplet'][int(2 if r['dis'] == 1 else 0)]}>"
                  f"{r['modified_triplet'][int(2 if r['dis'] == 1 else 0)]}",
        axis=1)
    df["triplet_transition"] = df["original_triplet"] + "->" + df["modified_triplet"]

    print("Group distribution:")
    for g in GROUP_ORDER:
        n = int((df["trip_group"] == g).sum())
        print(f"  {g:<22}: {n:,} ({100 * n / max(1, len(df)):.2f}%)")

    print(f"loss (delta S<0): {int((df['delta_score'] < 0).sum()):,}, "
          f"gain (delta S>0): {int((df['delta_score'] > 0).sum()):,}")
    return df


def plot_figure4e_boxplot(df):
    print("Plot Figure4E boxplot ...")
    sns.set_style("white")
    plt.rcParams.update({
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    a = df[df["trip_group"] == "GroupA"]["delta_score"].values
    bxp_list = []
    positions = []
    stats = _boxplot_stats(a)
    if stats is not None:
        stats["label"] = "GroupA"
        bxp_list.append(stats)
        positions.append(0)

    if bxp_list:
        bp = ax.bxp(
            bxp_list, positions=positions, widths=0.45,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.8),
            boxprops=dict(linewidth=1.0, edgecolor="#333"),
            whiskerprops=dict(linewidth=0.9, color="#333"),
            capprops=dict(linewidth=0.9, color="#333"),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE_DARK)
            patch.set_alpha(0.85)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks([0])
    ax.set_xticklabels(["GroupA"], fontsize=10.5)
    ax.set_ylabel("delta S = modifiedscore - editscore")

    mw_text = ""
    if len(a) >= 2:
        try:
            _, p = mannwhitneyu(a, np.zeros_like(a), alternative="two-sided")
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else "ns")
            mw_text = f"  |  MW p={p:.1e} {sig}"
        except Exception:
            pass
    ax.set_title(f"Figure4E. Boxplot: GroupA vs 0{mw_text}")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if bxp_list:
        whis_lo = min(s["whislo"] for s in bxp_list)
        whis_hi = max(s["whishi"] for s in bxp_list)
        span = whis_hi - whis_lo if whis_hi > whis_lo else 1e-3
        ax.set_ylim(whis_lo - span * 0.08, whis_hi + span * 0.18)

    fig.tight_layout()
    save_fig(fig, FIG_4E)


def plot_figure4f_top15(df):
    print("Plot Figure4F top15 triplet transitions ...")
    sns.set_style("white")
    plt.rcParams.update({
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if len(df) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.axis("off")
    else:
        agg = (
            df.groupby(["triplet_transition", "original_triplet"])
            .agg(n=("delta_score", "size"), med=("delta_score", "median"))
            .reset_index()
        )
        agg = agg[agg["n"] >= 100].copy()

        if len(agg) == 0:
            ax.text(0.5, 0.5, "no group with n>=100", ha="center", va="center")
            ax.axis("off")
        else:
            agg["abs_med"] = agg["med"].abs()
            top15 = agg.nlargest(15, "abs_med").sort_values("abs_med", ascending=True)
            colors = [BLUE_LIGHT] * len(top15)
            ax.barh(range(len(top15)), top15["med"].values,
                    color=colors, edgecolor="white", linewidth=1.2)

            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_yticks(range(len(top15)))
            ax.set_yticklabels(top15["triplet_transition"].tolist(),
                               fontsize=10, fontfamily="monospace")
            ax.set_xlabel("Median delta S")

            v = top15["med"].values
            lo, hi = v.min(), v.max()
            span = hi - lo if hi > lo else 1e-3
            ax.set_xlim(lo - span * 0.10, hi + span * 0.15)

    ax.set_title("Figure4F. Top 15 triplet transitions by |median delta S|")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, FIG_4F)


def export_transition_table(df):
    print(f"Export {os.path.basename(E_TRIPLET_SUMMARY)} ...")
    agg = (
        df.groupby(
            ["dis", "trip_group", "original_triplet",
             "modified_triplet", "triplet_transition"],
            observed=True,
        )
        .agg(
            n=("delta_score", "size"),
            med_dS=("delta_score", "median"),
            q25=("delta_score", lambda x: float(x.quantile(0.25))),
            q75=("delta_score", lambda x: float(x.quantile(0.75))),
            loss_frac=("delta_score", lambda x: float((x < 0).mean())),
            gain_frac=("delta_score", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    agg = agg.sort_values(["dis", "trip_group", "n"],
                          ascending=[True, True, False])
    agg.to_csv(E_TRIPLET_SUMMARY, sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {E_TRIPLET_SUMMARY} ({len(agg):,} rows)")


def run_pairwise_tests(df):
    print(f"Pairwise tests: GroupA vs GroupB (by dis) -> {os.path.basename(E_PAIRWISE_TESTS)}")
    results = []

    for dis in [1, -1]:
        sub = df[df["dis"] == dis]
        if len(sub) == 0:
            continue

        a = sub[sub["trip_group"] == "GroupA"]["delta_score"].values
        b = sub[sub["trip_group"] == "GroupB"]["delta_score"].values
        if len(a) < 100 or len(b) < 100:
            print(f"dis={dis:+d}: too few samples, skip")
            continue

        u, p = mannwhitneyu(a, b, alternative="two-sided")
        r = 1 - 2 * u / (len(a) * len(b))
        results.append({
            "dis": dis,
            "group1": "GroupA", "group2": "GroupB",
            "n1": len(a), "n2": len(b),
            "u_stat": float(u), "raw_p": float(p),
            "cliff_delta": float(r),
            "med1": float(np.median(a)), "med2": float(np.median(b)),
            "delta_median": float(np.median(a) - np.median(b)),
            "sig": ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "ns"),
        })

    if results:
        rdf = pd.DataFrame(results)
        rdf.to_csv(E_PAIRWISE_TESTS, sep="\t", index=False, float_format="%.6f")
        print(f"Saved: {E_PAIRWISE_TESTS}")
        for _, r in rdf.iterrows():
            print(f"  dis={r['dis']:+d}: delta median={r['delta_median']:+.5f}, "
                  f"Cliff's delta={r['cliff_delta']:+.3f}, "
                  f"p={r['raw_p']:.2e} {r['sig']}")


def main():
    np.random.seed(42)

    sampled, pos_full, pooled_full = load_figure4ab_data()

    pooled_full.to_csv(os.path.join(TEXT_DIR, "A_pooled_full.tsv"),
                       sep="\t", index=False, float_format="%.6f")
    pos_full.to_csv(os.path.join(TEXT_DIR, "B_pos_full.tsv"),
                    sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {os.path.join(TEXT_DIR, 'A_pooled_full.tsv')}")
    print(f"Saved: {os.path.join(TEXT_DIR, 'B_pos_full.tsv')}")

    stepA_res = step_A_abs_dis_trend(
        sampled, pooled_full, os.path.join(TEXT_DIR, "A"))
    plot_A_absdis_vs_median(pooled_full, stepA_res)
    plot_B_count_ratio_combined(pos_full)

    df_cd = load_cd_stats()
    plot_figure4c_4d(df_cd)

    plot_figure4c_heatmap()

    df_triplet = build_triplets()
    df_triplet = validate_triplets(df_triplet)
    df_triplet = preprocess(df_triplet)

    if df_triplet is None or len(df_triplet) == 0:
        print("No valid triplet data")
    else:
        plot_figure4e_boxplot(df_triplet)
        plot_figure4f_top15(df_triplet)
        export_transition_table(df_triplet)
        run_pairwise_tests(df_triplet)

    print("=" * 60)
    print("OUTPUT FILES:")
    for p in [
        FIG_4A_ABSDIS, FIG_4B_COMBINED, FIG_4C, FIG_4C_HEATMAP,
        FIG_4D, FIG_4E, FIG_4F,
        os.path.join(TEXT_DIR, "A_pooled_full.tsv"),
        os.path.join(TEXT_DIR, "A_stepA_abs_dis_trend.tsv"),
        os.path.join(TEXT_DIR, "B_pos_full.tsv"),
        STATS_C, STATS_D,
        E_DIS1_SUMMARY, E_DIS1_DUNN,
        E_TRIPLET_SUMMARY, E_PAIRWISE_TESTS,
        CENTER_MISMATCH_CSV, REF_MISMATCH_CSV,
    ]:
        print(f"  {p}")
    print("=" * 60)

    try:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"Cleaned temp dir: {TMP_DIR}")
    except Exception as e:
        print(f"WARN: failed to clean temp dir: {e}")


if __name__ == "__main__":
    main()