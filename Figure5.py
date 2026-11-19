#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 pipeline (A/B/C/D/E/F)

[Figure to output mapping]
    Figure 5A: Distribution of SNP prediction scores (bar) + SNP vs Edit donut
    Figure 5B: With-SNP vs Without-SNP donut + SNP count per site bar
    Figure 5C: Ridge plot of single best SNP per site (vs Reference)
    Figure 5D: |dis| distribution of best SNP (diff>0)
    Figure 5E: Ridge plot of Reference / SNP at dis+1 / double SNP
    Figure 5F: Residual rescue rate bar chart (multi SNP effect)

[External data (must be prepared in advance; fill paths in CONFIG['input'])]
    snp_predictions.bed  : SNP prediction scores (col 5 = score)
    edit_predictions.bed : Editing prediction scores
    snp_expanded.parquet : Expanded SNP table with columns
                           chrom,start,end,strand,snpindex,dis,editscore
    double_best.tsv      : Best double-SNP results per site
    multi_results.tsv    : Multi-SNP results per site
    dis1_tsv_dir         : directory with 4 category tsv files
                           (NonPref_to_Pref, Pref_to_Pref,
                            NonPref_to_NonPref, Pref_to_NonPref)

[Figure to file mapping]
    Figure 5A: Figure5/Figure5A.png
               Figure5/Figure5A_distribution.png
    Figure 5B: Figure5/Figure5B_donut.png
               Figure5/Figure5B_snp_per_site.png
    Figure 5C: Figure5/Figure5C_single_best.png
    Figure 5D: Figure5/Figure5D_bestSNP_dis_distribution.png
    Figure 5E: Figure5/Figure5E_ridge_double_dis1_ref.png
    Figure 5F: Figure5/Figure5F_residual_multi_stacked_bar.png
"""

import os
import sys
import gc

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, wilcoxon, mannwhitneyu


# ============================================================================
# CONFIG  (replace every path below with your own before running)
# ============================================================================
CONFIG = {
    "input": {
        "fig5a": {
            "snp_bed":  "/path/to/redi_overlap_results/predict_results/snp_predictions.bed",
            "edit_bed": "/path/to/redi_overlap_results/predict_results/edit_predictions.bed",
        },
        "fig5b": {
            "parquet": "/path/to/redi_overlap_results/with_SNP/snp_expanded.parquet",
            "n_total": 1495832,
        },
        "fig5c_d": {
            "parquet":  "/path/to/redi_overlap_results/with_SNP/snp_expanded.parquet",
            "snp_bed":  "/path/to/redi_overlap_results/with_SNP/predict_results/snp_predictions.bed",
            "edit_bed": "/path/to/redi_overlap_results/with_SNP/predict_results/edit_predictions.bed",
        },
        "fig5e_f": {
            "double_best_tsv": "/path/to/redi_overlap_results/dis_plus1_double_multi/double_best.tsv",
            "multi_tsv":       "/path/to/redi_overlap_results/dis_plus1_double_multi/multi_results.tsv",
            "dis1_tsv_dir":    "/path/to/redi_overlap_results/dis_plus1",
        },
    },
    "output": {
        "outdir": "/path/to/redi_overlap_results",
    },
}


# ============================================================================
# Derived paths
# ============================================================================
FIG5_DIR = os.path.join(CONFIG["output"]["outdir"], "Figure5")
FIG5C_D_STATS_DIR = os.path.join(FIG5_DIR, "single_bestsnp_stats")

OUT_5A_BAR    = os.path.join(FIG5_DIR, "Figure5A.png")
OUT_5A_DONUT  = os.path.join(FIG5_DIR, "Figure5A_distribution.png")
OUT_5B_DONUT  = os.path.join(FIG5_DIR, "Figure5B_donut.png")
OUT_5B_BAR    = os.path.join(FIG5_DIR, "Figure5B_snp_per_site.png")
OUT_5C_RIDGE  = os.path.join(FIG5_DIR, "Figure5C_single_best.png")
OUT_5D_DIS    = os.path.join(FIG5_DIR, "Figure5D_bestSNP_dis_distribution.png")
OUT_5E_RIDGE  = os.path.join(FIG5_DIR, "Figure5E_ridge_double_dis1_ref.png")
OUT_5F_BAR    = os.path.join(FIG5_DIR, "Figure5F_residual_multi_stacked_bar.png")

RIDGE_HEIGHT = 2.6
RIDGE_ASPECT = 4.0


# ============================================================================
# Common utilities
# ============================================================================
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def count_lines(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                n += 1
    return n


def format_pvalue(method, pval, n):
    if np.isnan(pval):
        return f'{method}: n/a (N={n:,})'
    if pval == 0 or pval < 2e-16:
        return f'{method} p < 2e-16 (N={n:,})'
    if pval < 1e-4:
        return f'{method} p = {pval:.3e} (N={n:,})'
    return f'{method} p = {pval:.4f} (N={n:,})'


# ============================================================================
# Figure 5A
# ============================================================================
def run_fig5a():
    print("\n" + "=" * 70)
    print("Figure 5A - SNP score distribution + SNP vs Edit donut")
    print("=" * 70)
    cfg = CONFIG["input"]["fig5a"]
    snp_bed  = cfg["snp_bed"]
    edit_bed = cfg["edit_bed"]
    check_file_exists(snp_bed)
    check_file_exists(edit_bed)
    os.makedirs(FIG5_DIR, exist_ok=True)

    # ---- (1) bar chart: distribution of SNP prediction scores ----
    scores = []
    with open(snp_bed) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 5:
                try:
                    scores.append(float(parts[4]))
                except ValueError:
                    pass

    bins = [round(0.05 * i, 2) for i in range(0, 10)]
    labels, counts = [], []
    for i in range(len(bins)):
        if i < len(bins) - 1:
            low, high = bins[i], bins[i + 1]
            labels.append(f"[{low:.2f},{high:.2f})")
            counts.append(sum(1 for s in scores if low <= s < high))
        else:
            low, high = bins[i], 0.50
            labels.append(f"[{low:.2f},{high:.2f}]")
            counts.append(sum(1 for s in scores if low <= s <= high))

    print("Score distribution:")
    for lab, cnt in zip(labels, counts):
        print(f"  {lab}: {cnt}")

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Score range')
    plt.ylabel('Count')
    plt.title('Distribution of SNP prediction scores (overlapping REDIportal)')
    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_5A_BAR, dpi=150)
    plt.close()
    print(f"Bar chart saved to {OUT_5A_BAR}")

    # ---- (2) donut chart: SNP vs Edit proportion ----
    n_snp  = count_lines(snp_bed)
    n_edit = count_lines(edit_bed)
    total  = n_snp + n_edit
    print(f"\nsnp_predictions.bed  : {n_snp:,}")
    print(f"edit_predictions.bed : {n_edit:,}")
    print(f"total                : {total:,}")

    values = [n_snp, n_edit]
    names  = ['snp_predictions', 'edit_predictions']
    colors = ['skyblue', '#7FD8BE']
    explode = [0.08, 0.0]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    wedges, _ = ax.pie(
        values, colors=colors, explode=explode,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2),
    )
    for w, name, v in zip(wedges, names, values):
        ang = (w.theta2 + w.theta1) / 2.0
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        pct = v / total * 100 if total else 0
        radius = 1.18 if name == 'snp_predictions' else 1.05
        ax.annotate(
            f"{name}\n{v:,} ({pct:.2f}%)",
            xy=(x * radius, y * radius),
            xytext=(x * (radius + 0.25), y * (radius + 0.25)),
            ha='left' if x >= 0 else 'right',
            va='center', fontsize=11,
            color='black' if name == 'snp_predictions' else '#2F7E68',
            fontweight='bold' if name == 'snp_predictions' else 'normal',
            arrowprops=dict(
                arrowstyle='-',
                color='black' if name == 'snp_predictions' else 'gray',
                lw=1.2 if name == 'snp_predictions' else 0.8,
            ),
        )
    ax.set_title('Proportion of SNP vs Edit predictions', pad=20)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUT_5A_DONUT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Donut chart saved to {OUT_5A_DONUT}")


# ============================================================================
# Figure 5B
# ============================================================================
def run_fig5b():
    print("\n" + "=" * 70)
    print("Figure 5B - With/Without SNP donut + SNP-per-site bar")
    print("=" * 70)
    cfg = CONFIG["input"]["fig5b"]
    parquet = cfg["parquet"]
    n_total = int(cfg["n_total"])
    check_file_exists(parquet)
    os.makedirs(FIG5_DIR, exist_ok=True)

    con = duckdb.connect()
    df_site = con.execute(f"""
        SELECT chrom, start, COUNT(*) AS n_snp
        FROM read_parquet('{parquet}')
        GROUP BY chrom, start
    """).df()

    n_with    = len(df_site)
    n_without = max(n_total - n_with, 0)
    print(f"N_total       : {n_total:,}")
    print(f"N_with        : {n_with:,} ({n_with/n_total*100:.4f}%)")
    print(f"N_without     : {n_without:,} ({n_without/n_total*100:.4f}%)")

    # ---- (1) donut ----
    names   = ['With SNP', 'Without SNP']
    values  = [n_with, n_without]
    colors  = ['skyblue', '#BFE5DD']
    explode = [0.08, 0.0]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    wedges, _ = ax.pie(
        values, colors=colors, explode=explode,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2),
    )
    for w, name, v in zip(wedges, names, values):
        ang = (w.theta2 + w.theta1) / 2.0
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        pct = v / n_total * 100 if n_total else 0
        r = 1.18 if name == 'With SNP' else 1.05
        ax.annotate(
            f"{name}\n{v:,} ({pct:.4f}%)",
            xy=(x * r, y * r),
            xytext=(x * (r + 0.30), y * (r + 0.30)),
            ha='left' if x >= 0 else 'right',
            va='center', fontsize=11,
            color='black' if name == 'With SNP' else '#2F7E68',
            fontweight='bold' if name == 'With SNP' else 'normal',
            arrowprops=dict(
                arrowstyle='-',
                color='black' if name == 'With SNP' else 'gray',
                lw=1.2 if name == 'With SNP' else 0.8,
            ),
        )
    ax.set_title('Original sites vs SNP-containing sites', pad=20)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUT_5B_DONUT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Donut chart saved to {OUT_5B_DONUT}")

    # ---- (2) SNP count per site bar (k >= 1) ----
    n_arr = df_site['n_snp'].astype(int).values
    max_n = int(n_arr.max()) if len(n_arr) > 0 else 0

    full_counts = np.zeros(max_n + 1, dtype=int)
    full_counts[0] = n_without
    for v in n_arr:
        full_counts[v] += 1

    print("\nSNP per site distribution (incl. k=0):")
    for k, c in enumerate(full_counts):
        print(f"  {k:>4} SNP : {c:>12,}  ({c / n_total * 100:7.4f}%)")

    k_vals = np.arange(1, len(full_counts))
    y_vals = full_counts[1:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(k_vals, y_vals, color='skyblue', edgecolor='white', width=0.9)
    ax.set_xlabel('Number of SNPs per original site')
    ax.set_ylabel('Number of sites')
    ax.set_title('Distribution of SNP counts per original site (k >= 1)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(left=0.4)

    tick_max = int(k_vals[-1]) if len(k_vals) > 0 else 1
    ticks = [1] + list(range(10, tick_max + 1, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

    plt.tight_layout()
    plt.savefig(OUT_5B_BAR, dpi=150)
    plt.close()
    print(f"Bar chart saved to {OUT_5B_BAR}")


# ============================================================================
# Figure 5C & 5D
# ============================================================================
def run_fig5c_d():
    print("\n" + "=" * 70)
    print("Figure 5C (ridge) + Figure 5D (dis distribution)")
    print("=" * 70)
    cfg = CONFIG["input"]["fig5c_d"]
    parquet  = cfg["parquet"]
    snp_bed  = cfg["snp_bed"]
    edit_bed = cfg["edit_bed"]
    check_file_exists(parquet)
    check_file_exists(snp_bed)
    check_file_exists(edit_bed)
    os.makedirs(FIG5C_D_STATS_DIR, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")

    print("[1] Loading prediction BEDs ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE pred AS
    WITH raw AS (
        SELECT * FROM read_csv('{snp_bed}', delim='\t', header=false, quote='',
            columns={{
                'chrom':'VARCHAR','start':'BIGINT','end':'BIGINT',
                'name':'VARCHAR','pred_score':'DOUBLE','strand':'VARCHAR'
            }})
        UNION ALL
        SELECT * FROM read_csv('{edit_bed}', delim='\t', header=false, quote='',
            columns={{
                'chrom':'VARCHAR','start':'BIGINT','end':'BIGINT',
                'name':'VARCHAR','pred_score':'DOUBLE','strand':'VARCHAR'
            }})
    )
    SELECT chrom, start, "end", strand, pred_score,
           regexp_extract(name, '^(snp|edit)_(.+)$', 2) AS snpindex,
           regexp_extract(name, '^(snp|edit)_(.+)$', 1) AS label
    FROM raw
    """)
    n_pred = con.execute("SELECT COUNT(*) FROM pred").fetchone()[0]
    print(f"    pred rows: {n_pred:,}")

    print("[2] Loading parquet ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE pq AS
    SELECT chrom, start, "end" AS end_pos, strand, snpindex, dis, editscore
    FROM read_parquet('{parquet}')
    """)
    n_pq = con.execute("SELECT COUNT(*) FROM pq").fetchone()[0]
    print(f"    parquet rows: {n_pq:,}")

    print("[3] Joining ...")
    con.execute("""
    CREATE OR REPLACE TABLE joined AS
    SELECT p.chrom, p.start, p."end", p.strand,
           p.snpindex, p.label, p.pred_score,
           pq.dis, pq.editscore
    FROM pred p
    JOIN pq
      ON p.chrom    = pq.chrom
     AND p.start    = pq.start
     AND p."end"    = pq.end_pos
     AND p.strand   = pq.strand
     AND p.snpindex = pq.snpindex
    """)
    n_joined = con.execute("SELECT COUNT(*) FROM joined").fetchone()[0]
    print(f"    joined rows: {n_joined:,}")

    print("[4] Collapsing to best SNP per site ...")
    con.execute("""
    CREATE OR REPLACE TABLE site_best AS
    SELECT * FROM (
        SELECT
            chrom, start, "end", strand,
            snpindex    AS best_snpindex,
            dis         AS best_dis,
            label       AS best_label,
            editscore,
            pred_score  AS max_pred_score,
            ROW_NUMBER() OVER (
                PARTITION BY chrom, start, "end", strand
                ORDER BY pred_score DESC, snpindex ASC
            ) AS rn,
            COUNT(*) OVER (
                PARTITION BY chrom, start, "end", strand
            ) AS n_snp_candidates
        FROM joined
    )
    WHERE rn = 1
    """)
    con.execute("ALTER TABLE site_best DROP COLUMN rn")
    con.execute("""
    CREATE OR REPLACE TABLE merged AS
    SELECT *, (max_pred_score - editscore) AS diff
    FROM site_best
    """)
    n_sites = con.execute("SELECT COUNT(*) FROM merged").fetchone()[0]
    print(f"    unique sites: {n_sites:,}")

    summary = con.execute("""
    SELECT
        COUNT(*)                                  AS n_total,
        SUM(CASE WHEN diff > 0 THEN 1 ELSE 0 END) AS n_pos,
        SUM(CASE WHEN diff < 0 THEN 1 ELSE 0 END) AS n_neg,
        SUM(CASE WHEN diff = 0 THEN 1 ELSE 0 END) AS n_zero,
        SUM(CASE WHEN diff > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_pos,
        AVG(diff)                                 AS mean_diff,
        MEDIAN(diff)                              AS median_diff,
        AVG(n_snp_candidates)                     AS mean_snp_per_site
    FROM merged
    """).df()
    print("\n=== Summary (best SNP per site) ===")
    print(summary.to_string(index=False))

    df_merged = con.execute("SELECT * FROM merged").df()
    df_merged.to_csv(os.path.join(FIG5C_D_STATS_DIR, "site_best_snp.tsv"),
                     sep='\t', index=False)
    summary.to_csv(os.path.join(FIG5C_D_STATS_DIR, "summary.tsv"),
                   sep='\t', index=False)

    # ---- Figure 5D: |dis| distribution of best SNP ----
    df_pos = df_merged[df_merged['diff'] > 0].copy()
    dis_vals = df_pos['best_dis'].dropna().astype(int).values
    dis_counts_pos = pd.Series(dis_vals[dis_vals > 0]).value_counts().sort_index()
    dis_counts_neg = pd.Series(np.abs(dis_vals[dis_vals < 0])).value_counts().sort_index()

    all_dis = sorted(set(dis_counts_pos.index).union(set(dis_counts_neg.index)))
    if all_dis:
        dmax = max(all_dis)
        grid = list(range(1, dmax + 1))
    else:
        grid = []

    pos_y = [int(dis_counts_pos.get(d, 0)) for d in grid]
    neg_y = [int(dis_counts_neg.get(d, 0)) for d in grid]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.4
    x = np.arange(len(grid))
    ax.bar(x - width / 2, neg_y, width=width, color="#A8D8EA",
           edgecolor='white', label='Upstream (-dis)')
    ax.bar(x + width / 2, pos_y, width=width, color="#FFD9A0",
           edgecolor='white', label='Downstream (+dis)')

    tick_positions = [i for i, d in enumerate(grid)
                      if d == 1 or d == 5 or d % 10 == 0]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(grid[i]) for i in tick_positions])

    ax.set_xlabel('|dis| of the best SNP (bp from editing site)')
    ax.set_ylabel('Number of sites')
    ax.set_title(f'Figure5D - |dis| of best SNP per site (diff>0, N={len(df_pos):,})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_5D_DIS, dpi=150)
    plt.close()
    print(f"Saved: {OUT_5D_DIS}")

    # ---- Figure 5C: ridge plot (single best SNP vs Reference) ----
    es = df_pos['editscore'].values
    ms = df_pos['max_pred_score'].values
    print(f"\nN (diff>0) = {len(df_pos):,}")
    print(f"mean editscore  = {np.nanmean(es):.4f}")
    print(f"mean best SNP   = {np.nanmean(ms):.4f}")

    names_list  = ["Reference", "Best SNP"]
    colors_list = ["#B5EAD7", "#FFD9A0"]
    values_list = [es, ms]

    data = []
    for vals, name in zip(values_list, names_list):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        for v in vals:
            data.append({'Score type': name, 'Score': v})
    long_df = pd.DataFrame(data)
    long_df['Score type'] = pd.Categorical(long_df['Score type'],
                                           categories=names_list, ordered=True)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(long_df, row="Score type", hue="Score type",
                      aspect=4.0, height=2.6, palette=colors_list)

    def ridge_plot(x, **kwargs):
        ax = plt.gca()
        color = kwargs.pop('color', 'steelblue')
        kde = gaussian_kde(x, bw_method=0.25)
        xgrid = np.linspace(max(x.min(), 0.0), min(x.max(), 1.0), 400)
        ygrid = kde(xgrid)
        ax.fill_between(xgrid, ygrid, alpha=0.75, color=color)
        ax.plot(xgrid, ygrid, lw=1.2, color='white')
        ax.axvline(x.mean(), color=color, linestyle='--',
                   linewidth=1.5, alpha=0.9, zorder=5)

    g.map(ridge_plot, "Score")
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(left=True, bottom=False)
    g.set(xlim=(0, 1))
    g.figure.subplots_adjust(left=0.22, hspace=-0.45)

    for ax, label in zip(g.axes.flat, names_list):
        ax.text(x=-0.02, y=0.2, s=label, transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                ha="right", va="center", clip_on=False)

    va = np.asarray(values_list[0], dtype=float); va = va[~np.isnan(va)]
    vb = np.asarray(values_list[1], dtype=float); vb = vb[~np.isnan(vb)]
    try:
        if len(va) == len(vb):
            stat, pval = wilcoxon(va, vb); method = 'Wilcoxon'
        else:
            stat, pval = mannwhitneyu(va, vb, alternative='two-sided')
            method = 'Mann-Whitney'
    except Exception as e:
        pval = np.nan
        method = f'test ({type(e).__name__})'

    ptext = format_pvalue(method, pval, len(va))
    mean_lines = [
        f"mean {names_list[i]} = "
        f"{np.nanmean(np.asarray(values_list[i], dtype=float)):.4f}"
        for i in range(len(names_list))
    ]
    box_text = ptext + "\n" + "\n".join(mean_lines)
    ax0 = g.axes.flat[0]
    ax0.text(0.98, 0.98, box_text, transform=ax0.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.35', fc='white',
                       ec='gray', lw=0.8, alpha=0.9))

    plt.xlabel("Score", fontsize=11)
    plt.suptitle(f"Best single SNP per site (diff>0)  (N = {len(va):,})",
                 y=1.02, fontsize=12)
    plt.savefig(OUT_5C_RIDGE, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved: {OUT_5C_RIDGE}")


# ============================================================================
# Figure 5E & 5F
# ============================================================================
def _load_dis1_all(dis1_tsv_dir):
    cat_tags = ["NonPref_to_Pref", "Pref_to_Pref",
                "NonPref_to_NonPref", "Pref_to_NonPref"]
    dis1_list = []
    for tag in cat_tags:
        fp = os.path.join(dis1_tsv_dir, f"{tag}.tsv")
        if os.path.exists(fp):
            tmp = pd.read_csv(fp, sep="\t")
            tmp["site_group"] = tmp.apply(
                lambda r: f"{r.chrom}:{r.start}-{r.end}({r.strand})", axis=1)
            dis1_list.append(tmp)
        else:
            print(f"Warning: file not found {fp}", file=sys.stderr)
    if len(dis1_list) == 0:
        raise FileNotFoundError(f"No dis+1 tsv files found in {dis1_tsv_dir}")
    return pd.concat(dis1_list)


def _load_part1_all_dis1(dis1_tsv_dir):
    print("\n[Part1] All dis+1 SNP sites (require ref & dis1 score non-null)")
    df_dis1_all = _load_dis1_all(dis1_tsv_dir)
    df_dis1_all = df_dis1_all.rename(
        columns={"modifiedscore": "score_dis1", "editscore": "score_ref"})
    df_part1 = df_dis1_all.dropna(subset=["score_ref", "score_dis1"]).copy()
    total = len(df_part1)

    n_gt = int((df_part1["score_dis1"] >  df_part1["score_ref"]).sum())
    n_lt = int((df_part1["score_dis1"] <= df_part1["score_ref"]).sum())
    pct_gt = n_gt / total * 100 if total else 0
    pct_lt = n_lt / total * 100 if total else 0

    mean_ref  = df_part1["score_ref"].mean()
    mean_dis1 = df_part1["score_dis1"].mean()
    try:
        _, pval = wilcoxon(df_part1["score_ref"], df_part1["score_dis1"])
        method = "Wilcoxon signed-rank"
    except Exception as e:
        pval = np.nan
        method = f"test failed({type(e).__name__})"

    print("\n========== Part1: All dis+1 SNP sites ==========")
    print(f"Total dis+1 sites            : {total:,}")
    print(f"dis+1 >  ref                 : {n_gt:,} ({pct_gt:.2f}%)")
    print(f"dis+1 <= ref                 : {n_lt:,} ({pct_lt:.2f}%)")
    print(f"Mean Reference               : {mean_ref:.4f}")
    print(f"Mean dis+1 SNP               : {mean_dis1:.4f}")
    print(f"{method}                    : {format_pvalue('', pval, total).strip()}")
    return df_part1


def _load_part2_three_scores(dis1_tsv_dir, double_best_tsv):
    print("\n[Part2] Sites with ref / dis1 / double all non-null")
    if not os.path.exists(double_best_tsv):
        raise FileNotFoundError(f"{double_best_tsv} not found")
    df_double = pd.read_csv(double_best_tsv, sep="\t")

    cat_tags = ["NonPref_to_Pref", "Pref_to_Pref",
                "NonPref_to_NonPref", "Pref_to_NonPref"]
    dis1_list = []
    for tag in cat_tags:
        fp = os.path.join(dis1_tsv_dir, f"{tag}.tsv")
        if os.path.exists(fp):
            tmp = pd.read_csv(fp, sep="\t")
            tmp["site_group"] = tmp.apply(
                lambda r: f"{r.chrom}:{r.start}-{r.end}({r.strand})", axis=1)
            dis1_list.append(tmp[["site_group", "modifiedscore"]])
        else:
            print(f"Warning: file not found {fp}", file=sys.stderr)
    if len(dis1_list) == 0:
        raise FileNotFoundError(f"No dis+1 tsv files found in {dis1_tsv_dir}")
    df_dis1 = pd.concat(dis1_list).rename(columns={"modifiedscore": "score_dis1"})

    df_merge = pd.merge(df_double, df_dis1, on="site_group", how="inner")
    df_merge = df_merge.rename(
        columns={"editscore": "score_ref", "pred_score": "score_double"})
    df_part2 = df_merge.dropna(
        subset=["score_ref", "score_dis1", "score_double"]).copy()
    total = len(df_part2)

    subA = df_part2[df_part2["score_dis1"] > df_part2["score_ref"]]
    n_subA = len(subA)
    n_subA_further = int((subA["score_double"] > subA["score_dis1"]).sum())
    pct_subA = n_subA_further / n_subA * 100 if n_subA else 0

    subB = df_part2[df_part2["score_dis1"] < df_part2["score_ref"]]
    n_subB = len(subB)
    n_subB_recover = int((subB["score_double"] > subB["score_ref"]).sum())
    pct_subB = n_subB_recover / n_subB * 100 if n_subB else 0

    mean_ref    = df_part2["score_ref"].mean()
    mean_dis1   = df_part2["score_dis1"].mean()
    mean_double = df_part2["score_double"].mean()
    try:
        _, pval = wilcoxon(df_part2["score_dis1"], df_part2["score_double"])
        method = "Wilcoxon signed-rank"
    except Exception as e:
        pval = np.nan
        method = f"test failed({type(e).__name__})"

    print("\n========== Part2: dis+1 + second SNP ==========")
    print(f"Total sites with all 3 scores : {total:,}")
    print(f"Subset A (dis+1 > ref)        : {n_subA:,}")
    print(f"    double > dis+1            : {n_subA_further:,} ({pct_subA:.2f}%)")
    print(f"Subset B (dis+1 < ref)        : {n_subB:,}")
    print(f"    double > ref              : {n_subB_recover:,} ({pct_subB:.2f}%)")
    print(f"Mean Reference                : {mean_ref:.4f}")
    print(f"Mean dis+1 SNP                : {mean_dis1:.4f}")
    print(f"Mean double SNP               : {mean_double:.4f}")
    print(f"{method} (dis1 vs double)   : {format_pvalue('', pval, total).strip()}")
    return df_part2


def _plot_ridge_comparison(values_list, names_list, colors_list,
                           title, out_png, xlim=(0, 1)):
    data = []
    for vals, name in zip(values_list, names_list):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        for v in vals:
            data.append({'Score type': name, 'Score': v})
    long_df = pd.DataFrame(data)
    if len(long_df) < 20:
        print(f"      skip {out_png}: too few points")
        return

    long_df['Score type'] = pd.Categorical(long_df['Score type'],
                                           categories=names_list, ordered=True)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(long_df, row="Score type", hue="Score type",
                      aspect=RIDGE_ASPECT, height=RIDGE_HEIGHT,
                      palette=colors_list)

    def ridge_plot(x, **kwargs):
        ax = plt.gca()
        color = kwargs.pop('color', 'steelblue')
        kde = gaussian_kde(x, bw_method=0.25)
        xgrid = np.linspace(max(x.min(), xlim[0]), min(x.max(), xlim[1]), 400)
        ygrid = kde(xgrid)
        ax.fill_between(xgrid, ygrid, alpha=0.75, color=color)
        ax.plot(xgrid, ygrid, lw=1.2, color='white')
        ax.axvline(x.mean(), color=color, linestyle='--',
                   linewidth=1.5, alpha=0.9, zorder=5)

    g.map(ridge_plot, "Score")
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(left=True, bottom=False)
    g.set(xlim=xlim)
    g.figure.subplots_adjust(left=0.22, hspace=-0.45)
    for ax, label in zip(g.axes.flat, names_list):
        ax.text(x=-0.02, y=0.2, s=label, transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                ha="right", va="center", clip_on=False)

    va = np.asarray(values_list[0], dtype=float); va = va[~np.isnan(va)]
    vb = np.asarray(values_list[1], dtype=float); vb = vb[~np.isnan(vb)]
    try:
        if len(va) == len(vb):
            stat, pval = wilcoxon(va, vb); method = 'Wilcoxon'
        else:
            stat, pval = mannwhitneyu(va, vb, alternative='two-sided')
            method = 'Mann-Whitney'
    except Exception as e:
        pval = np.nan
        method = f'test ({type(e).__name__})'

    ptext = format_pvalue(method, pval, len(va))
    mean_lines = [
        f"mean {names_list[i]} = "
        f"{np.nanmean(np.asarray(values_list[i], dtype=float)):.4f}"
        for i in range(len(names_list))
    ]
    box_text = ptext + "\n" + "\n".join(mean_lines)
    ax0 = g.axes.flat[0]
    ax0.text(0.98, 0.98, box_text, transform=ax0.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.35', fc='white',
                       ec='gray', lw=0.8, alpha=0.9))

    plt.xlabel("Editing score", fontsize=11)
    plt.suptitle(f"{title}  (N = {len(va):,})", y=1.02, fontsize=12)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"      saved: {out_png}")


def _plot_rescue_rate_bar(n_total, n_rescued, out_png):
    from matplotlib.patches import Patch
    n_failed = n_total - n_rescued
    pct = n_rescued / n_total * 100 if n_total else 0
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.barh(0, n_rescued, color="#D5B9F0", edgecolor='white', height=0.55)
    ax.barh(0, n_failed, left=n_rescued, color="#E0E0E0",
            edgecolor='white', height=0.55)
    ax.text(n_rescued / 2, 0, f"{pct:.2f}%",
            ha='center', va='center', fontsize=14,
            fontweight='bold', color='#4A2E6B')
    ax.text(n_rescued + n_failed / 2, 0, f"{100-pct:.2f}%",
            ha='center', va='center', fontsize=11, color='#555555')
    ax.set_yticks([])
    ax.set_xlabel('Number of residual sites')
    ax.set_title(f'Residual, n = {n_total:,}  |  multi SNP rescued, n = {n_rescued:,}',
                 fontsize=11)
    handles = [
        Patch(facecolor="#D5B9F0", edgecolor='white',
              label=f'multi SNP rescued, n = {n_rescued:,} ({pct:.2f}%)'),
        Patch(facecolor="#E0E0E0", edgecolor='white',
              label=f'still failed, n = {n_failed:,} ({100-pct:.2f}%)'),
    ]
    ax.legend(handles=handles, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.55), ncol=2, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      saved: {out_png}")


def run_fig5e_f():
    print("\n" + "=" * 70)
    print("Figure 5E (ridge) + Figure 5F (residual rescue bar)")
    print("=" * 70)
    cfg = CONFIG["input"]["fig5e_f"]
    double_best_tsv = cfg["double_best_tsv"]
    multi_tsv       = cfg["multi_tsv"]
    dis1_tsv_dir    = cfg["dis1_tsv_dir"]
    check_file_exists(double_best_tsv)
    check_file_exists(multi_tsv)
    os.makedirs(FIG5_DIR, exist_ok=True)

    _ = _load_part1_all_dis1(dis1_tsv_dir)
    df_p2 = _load_part2_three_scores(dis1_tsv_dir, double_best_tsv)

    # ---- Figure 5E: ridge plot (double > ref subset) ----
    df_plot = df_p2[df_p2["score_double"] > df_p2["score_ref"]].copy()
    print(f"\n[Plot E] sites with double > ref: {len(df_plot):,}")
    if len(df_plot) > 0:
        _plot_ridge_comparison(
            [df_plot['score_ref'].values,
             df_plot['score_dis1'].values,
             df_plot['score_double'].values],
            ["Reference", "SNP at dis+1", "double SNP"],
            ["#B5EAD7", "#A8D8EA", "#D5B9F0"],
            "Distribution of editing score (sites with double > ref)",
            OUT_5E_RIDGE,
        )

    # ---- Figure 5F: residual rescue rate bar ----
    print("\n[Plot F] Residual: dis1<=ref AND double<=ref; rescue = multi > ref")
    subB = df_p2[df_p2["score_dis1"] <= df_p2["score_ref"]].copy()
    resid = subB[subB["score_double"] <= subB["score_ref"]].copy()
    print(f"  Step1 dis1<=ref (subset B): {len(subB):,}")
    print(f"  Step2 double<=ref (Residual): {len(resid):,}")
    if len(resid) == 0:
        print("  skip bar: no residual sites")
        return

    df_multi = pd.read_csv(multi_tsv, sep="\t")[["site_group", "pred_score"]]
    df_multi = (df_multi.groupby("site_group", as_index=False)["pred_score"].max()
                        .rename(columns={"pred_score": "score_multi"}))

    df = resid.merge(df_multi, on="site_group", how="inner")
    df = df.dropna(subset=["score_multi"])
    print(f"  Residual with multi score: {len(df):,}")

    n_total   = len(df)
    n_rescued = int((df["score_multi"] > df["score_ref"]).sum())
    print(f"  multi > ref (rescued): {n_rescued:,} "
          f"({n_rescued / n_total * 100:.2f}%)")

    _plot_rescue_rate_bar(n_total, n_rescued, OUT_5F_BAR)


# ============================================================================
# Main
# ============================================================================
def main():

    print("=" * 70)
    print("Figure 5 pipeline - steps: all")
    print(f"Output root: {CONFIG['output']['outdir']}")
    print("=" * 70)

    os.makedirs(FIG5_DIR, exist_ok=True)

    try:
        run_fig5a()
        run_fig5b()
        run_fig5c_d()
        run_fig5e_f()
        print("\n=== All Figure 5 panels completed ===")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()