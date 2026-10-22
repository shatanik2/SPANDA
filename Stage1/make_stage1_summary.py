#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 1 summary: builds the summary CSV from the per-star logs of
# stage1_background_fit.py (only completed runs), prints the diagnostic and
# residual-quality tiers, writes them as an AASTeX table, and makes the runtime
# histograms and the diagnostics binned in numax (MAP) and in
# dnu_input / (0.263 numax_input^0.772).
# Output: ./stage1_summary_<N>/
# Usage:  python3 make_stage1_summary.py --logdir <DIR> [--catalog <CSV>] [--mission kepler --cadence LC]
# ============================================================
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, MaxNLocator,
                               NullFormatter, ScalarFormatter)

PARAM_PREFIXES = [
    "nu_rmmax", "log_10_a_1_over_b_1", "log_10_b_1", "log_10_a_2", "log_10_a_3",
    "log_10_b_2", "log_10_b_3", "log_10_W", "log_10_H_env", "HWHM",
]

EXPECTED_LABELS = [
    r"$\nu_{\rm{max}}$", r"$\log_{10} (a_1/b_1)$", r"$\log_{10} b_1$", r"$\log_{10} a_2$",
    r"$\log_{10} a_3$", r"$\log_{10} b_2$", r"$\log_{10} b_3$", r"$\log_{10} W$",
    r"$\log_{10} H_{env}$", r"$\mathrm{HWHM}$",
]

HEADER = [
    "kic", "numax_input", "dnu_input",
    "de_time_sec", "de_peak_nll", "hmc_time_sec", "total_runtime_min",
    "max_r_hat", "min_effective_samples", "divergent_transitions", "convergence_met",
]
for p in PARAM_PREFIXES:
    HEADER += [f"{p}_val", f"{p}_err_pos", f"{p}_err_neg", f"{p}_map", f"{p}_in_1sigma"]
HEADER += [
    "mean_res_full", "std_res_full",
    "mean_res_outside", "std_res_outside",
    "mean_res_inside", "std_res_inside",
]

NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"

RE_TOTAL   = re.compile(r"Total runtime:\s*" + NUM + r"\s*minutes")
RE_INPUT   = re.compile(r"KIC=(\d+)\s*\|\s*numax=" + NUM + r"\s*\S*\s*\|\s*Dnu=" + NUM)
RE_DE      = re.compile(r"DE Finished in\s*" + NUM + r"\s*sec\.\s*Peak NLL:\s*" + NUM)
RE_HMC     = re.compile(r"HMC Finished in\s*" + NUM + r"\s*sec\.")
RE_RHAT    = re.compile(r"(?:Max split R-hat \(single chain\)|Max Gelman-Rubin r_hat)\s*:\s*" + NUM)
RE_NEFF    = re.compile(r"Min Effective Samples\s*:\s*" + NUM)
RE_DIV     = re.compile(r"Divergent Transitions\s*:\s*" + NUM)
RE_SUCCESS = re.compile(r"^SUCCESS: HMC chain has", re.M)
RE_WARNING = re.compile(r"^WARNING: Theoretical convergence criteria not fully met", re.M)
RE_UNC     = re.compile(r"\+\s*" + NUM + r"\s*/\s*-\s*" + NUM)
RE_RES_FULL = re.compile(r"Mean of residuals \([^)]*full region\):\s*" + NUM + r"\s*\|\s*Std dev:\s*" + NUM)
RE_RES_OUT  = re.compile(r"Mean of residuals \(Outside\):\s*" + NUM + r"\s*\|\s*Std dev:\s*" + NUM)
RE_RES_IN   = re.compile(r"Mean of residuals \(Inside\):\s*" + NUM + r"\s*\|\s*Std dev:\s*" + NUM)
RE_LOGNAME  = re.compile(r"kic_(\d+)\.log$")


def require(regex, text, what):
    m = regex.search(text)
    if m is None:
        raise ValueError(f"missing {what}")
    return m.groups()


def parse_param_table(text):
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "--- HMC Inferred Parameters ---")
    except StopIteration:
        raise ValueError("missing parameter table")
    rows = lines[start + 3:start + 3 + len(PARAM_PREFIXES)]
    if len(rows) != len(PARAM_PREFIXES):
        raise ValueError("incomplete parameter table")

    values = []
    for row, expected in zip(rows, EXPECTED_LABELS):
        parts = [p.strip() for p in row.split("|")]
        if len(parts) != 5:
            raise ValueError(f"malformed table row: {row!r}")
        label, median, unc, map_val, in_1sigma = parts
        if label != expected:
            raise ValueError(f"unexpected parameter {label!r} (expected {expected!r})")
        m = RE_UNC.search(unc)
        if m is None:
            raise ValueError(f"malformed uncertainty: {unc!r}")
        if in_1sigma not in ("True", "False"):
            raise ValueError(f"malformed In 1-Sigma value: {in_1sigma!r}")
        values += [median, m.group(1), m.group(2), map_val, in_1sigma]
    return values


def parse_log(path, catalogue):
    text = path.read_text(encoding="utf-8", errors="replace")

    total = RE_TOTAL.search(text)
    if total is None:
        return None, "incomplete"

    kic, numax_log, dnu_log = require(RE_INPUT, text, "KIC/numax/Dnu line")
    if catalogue is not None and int(kic) in catalogue:
        numax_in, dnu_in = catalogue[int(kic)]
    else:
        numax_in, dnu_in = numax_log, dnu_log

    de_time, de_nll = require(RE_DE, text, "DE timing line")
    (hmc_time,) = require(RE_HMC, text, "HMC timing line")
    (r_hat,) = require(RE_RHAT, text, "R-hat line")
    (n_eff,) = require(RE_NEFF, text, "effective samples line")
    (div,) = require(RE_DIV, text, "divergences line")

    if RE_SUCCESS.search(text):
        converged = "True"
    elif RE_WARNING.search(text):
        converged = "False"
    else:
        raise ValueError("missing convergence verdict")

    params = parse_param_table(text)
    res_full = require(RE_RES_FULL, text, "full-region residuals line")
    res_out = require(RE_RES_OUT, text, "outside residuals line")
    res_in = require(RE_RES_IN, text, "inside residuals line")

    row = [kic, numax_in, dnu_in, de_time, de_nll, hmc_time, total.group(1),
           r_hat, n_eff, div, converged] + params + list(res_full + res_out + res_in)
    return row, "ok"


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def np_isfinite(x):
    return x == x and x not in (float("inf"), float("-inf"))


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def diagnostic_tiers(rows):
    """Stars per diagnostic tier; non-finite values go to the worst tier."""
    i_rhat = HEADER.index("max_r_hat")
    i_neff = HEADER.index("min_effective_samples")
    i_div = HEADER.index("divergent_transitions")
    i_mu = HEADER.index("mean_res_outside")
    i_sd = HEADER.index("std_res_outside")

    counts = {k: 0 for k in [
        "rhat_strong", "rhat_standard", "rhat_unconv",
        "neff_opt", "neff_border", "neff_severe",
        "div_zero", "div_mild", "div_path",
        "res_pass", "res_fail",
    ]}
    nonfinite = {"max_r_hat": 0, "min_effective_samples": 0, "divergent_transitions": 0, "residuals_outside": 0}

    for row in rows.values():
        r = to_float(row[i_rhat])
        if not np_isfinite(r):
            nonfinite["max_r_hat"] += 1
            counts["rhat_unconv"] += 1
        elif r <= 1.01:
            counts["rhat_strong"] += 1
        elif r <= 1.05:
            counts["rhat_standard"] += 1
        else:
            counts["rhat_unconv"] += 1

        n = to_float(row[i_neff])
        if not np_isfinite(n):
            nonfinite["min_effective_samples"] += 1
            counts["neff_severe"] += 1
        elif n >= 400:
            counts["neff_opt"] += 1
        elif n >= 300:
            counts["neff_border"] += 1
        else:
            counts["neff_severe"] += 1

        d = to_float(row[i_div])
        if not np_isfinite(d):
            nonfinite["divergent_transitions"] += 1
            counts["div_path"] += 1
        elif d == 0:
            counts["div_zero"] += 1
        elif d <= 30:
            counts["div_mild"] += 1
        else:
            counts["div_path"] += 1

        mu, sd = to_float(row[i_mu]), to_float(row[i_sd])
        if not (np_isfinite(mu) and np_isfinite(sd)):
            nonfinite["residuals_outside"] += 1
            counts["res_fail"] += 1
        elif abs(1.0 - mu) <= 0.02 + 1e-9 and abs(1.0 - sd) <= 0.05 + 1e-9:
            counts["res_pass"] += 1
        else:
            counts["res_fail"] += 1

    return counts, nonfinite


def print_diagnostics(counts, nonfinite, total):
    groups = [
        ("max(split R_hat)", [
            ("<= 1.01", counts["rhat_strong"]),
            ("1.01 < max(R_hat) <= 1.05", counts["rhat_standard"]),
            ("> 1.05", counts["rhat_unconv"]),
        ]),
        ("min(N_eff)", [
            (">= 400", counts["neff_opt"]),
            ("300 <= min(N_eff) < 400", counts["neff_border"]),
            ("< 300", counts["neff_severe"]),
        ]),
        ("Divergent Transitions", [
            ("= 0", counts["div_zero"]),
            ("1 <= Divergences <= 30", counts["div_mild"]),
            ("> 30", counts["div_path"]),
        ]),
        ("Fractional Residual Quality (outside numax* +/- 3.5 HWHM*)", [
            ("Passed (|1 - mu_res,out| <= 0.02, |1 - sigma_res,out| <= 0.05)", counts["res_pass"]),
            ("Fractional Residual Outliers", counts["res_fail"]),
        ]),
    ]
    print("\n--- MCMC Diagnostics and Fractional Residual Quality Tiers ---")
    print(f"Total stars: N = {total:,}")
    for name, tiers in groups:
        print(f"\n{name}")
        for label, c in tiers:
            print(f"  {label:<66} {c:>8,}  {pct(c, total):6.2f} %")
    for key, c in nonfinite.items():
        if c:
            print(f"Note: {c:,} star(s) have a non-finite {key}; counted in the worst tier.")


def write_latex_table(counts, total, out_path):
    def line(count):
        return f"{count:,} & {pct(count, total):.2f}"

    n_str = f"{total:,}".replace(",", "{,}")
    tex = rf"""\begin{{deluxetable*}}{{llcc}}
\tablecaption{{MCMC Sampling Diagnostics and Fractional Residual Quality Tier Breakdown for \texttt{{SPANDA-Stage 1}}\label{{tab:spanda_stage1_diagnostics}}}}
\tablehead{{\colhead{{Parameter}} & \colhead{{Zone / Diagnostic Criterion}} & \colhead{{Star Count}} & \colhead{{Percentage (\%)}}}}
\startdata
$\max(\hat{{R}})$ & $\leq 1.01$ & {line(counts["rhat_strong"])} \\
 & $1.01 < \max(\hat{{R}}) \leq 1.05$ & {line(counts["rhat_standard"])} \\
 & $> 1.05$ & {line(counts["rhat_unconv"])} \\
\hline
$\min(N_{{\rm eff}})$ & $\geq 400$ & {line(counts["neff_opt"])} \\
 & $300 \leq \min(N_{{\rm eff}}) < 400$ & {line(counts["neff_border"])} \\
 & $< 300$ & {line(counts["neff_severe"])} \\
\hline
Divergent Transitions & $= 0$ & {line(counts["div_zero"])} \\
 & $1 \leq$ Divergences $\leq 30$ & {line(counts["div_mild"])} \\
 & $> 30$ & {line(counts["div_path"])} \\
\hline
Fractional Residual Quality & Passed ($|1-\mu_{{\rm res,out}}| \leq 0.02,\ |1-\sigma_{{\rm res,out}}| \leq 0.05$) & {line(counts["res_pass"])} \\
(Outside, $|\nu - \nu^{{*}}_{{\rm max}}| > 3.5\,{{\rm HWHM}}^{{*}}$) & Fractional Residual Outliers & {line(counts["res_fail"])} \\
\enddata
\tablecomments{{Total stars: $N = {n_str}$. Here, $\nu^{{*}}_{{\rm max}}$ and ${{\rm HWHM}}^{{*}}$ are the MAP values of $\nu_{{\rm max}}$ and HWHM, and $\mu_{{\rm res,out}}$ and $\sigma_{{\rm res,out}}$ refer to the mean and the standard deviation of the fractional residuals computed outside the $\nu^{{*}}_{{\rm max}} \pm 3.5\,{{\rm HWHM}}^{{*}}$ range. All percentages are relative to the full sample.}}
\end{{deluxetable*}}
"""
    with open(out_path, "w") as f:
        f.write(tex)


# ============================================================
# Runtime histograms
# ============================================================
BAR_FACE, BAR_EDGE = "#A7A6EA", "#8786DD"
MEAN_C, MEDIAN_C, MINUS1S_C, PLUS1S_C = "#008080", "#E06653", "#000000", "#B0307C"
LABEL_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE = 16, 14, 14
N_BINS = 60


def plot_hist(pdf, data, xlabel):
    mean = np.mean(data)
    p16, p50, p84 = np.percentile(data, [15.87, 50.0, 84.13])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(data, bins=N_BINS, color=BAR_FACE, edgecolor=BAR_EDGE, linewidth=0.6)
    ax.axvline(mean, color=MEAN_C, linestyle="--", lw=2.5, label=f"Mean: {mean:.2f} s")
    ax.axvline(p50, color=MEDIAN_C, linestyle="--", lw=2.5, label=f"Median (50.00%): {p50:.2f} s")
    ax.axvline(p16, color=MINUS1S_C, linestyle=":", lw=2.5, label=rf"$-1\sigma$ (15.87%): {p16:.2f} s")
    ax.axvline(p84, color=PLUS1S_C, linestyle="-.", lw=2.5, label=rf"$+1\sigma$ (84.13%): {p84:.2f} s")

    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Count (Number of Stars)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, edgecolor="0.3")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return mean, p16, p50, p84


def column(rows, name):
    i = HEADER.index(name)
    return np.array([to_float(r[i]) for r in rows.values()], dtype=float)


def make_runtime_histograms(rows, outdir):
    de = column(rows, "de_time_sec")
    hmc = column(rows, "hmc_time_sec")
    ok = np.isfinite(de) & np.isfinite(hmc)
    de, hmc = de[ok], hmc[ok]
    n_used = len(de)
    if n_used == 0:
        print("No valid DE/HMC times found; runtime histograms not made.")
        return ["Runtime histograms: not made (no valid DE/HMC times)."]

    out_path = Path(outdir) / f"Stage1_DE_HMC_times_{n_used}.pdf"
    with PdfPages(out_path) as pdf:
        s_de = plot_hist(pdf, de, "DE Time (seconds)")
        s_hmc = plot_hist(pdf, hmc, "HMC Time (seconds)")
    print(f"\nStars with valid DE and HMC times: {n_used:,}")
    print(f"Saved runtime histograms: {out_path.name}")

    lines = [f"[{out_path.name}]",
             f"  Stars used: {n_used:,} (finite DE and HMC times)",
             f"  Binning: {N_BINS} equal-width bins on a linear x-axis; log y-axis",
             "  Lines: mean; median (50th); 15.87th and 84.13th percentiles"]
    for page, (name, (mean, p16, p50, p84)) in enumerate(
            [("DE time (s)", s_de), ("HMC time (s)", s_hmc)], start=1):
        lines.append(f"  Page {page}: {name}: mean = {mean:.2f}, median = {p50:.2f}, "
                     f"15.87th = {p16:.2f}, 84.13th = {p84:.2f}")
    return lines


# ============================================================
# Diagnostics binned in numax (MAP) and in the dnu-numax ratio
# ============================================================
DIAG_FIGSIZE = (7.0, 7.4)
DIAG_LABEL_FONTSIZE = 10.5
DIAG_TICK_FONTSIZE = 9
DIAG_LEGEND_FONTSIZE = 9.5
DIAG_PANEL_FONTSIZE = 9
DIAG_COLUMN_GAP = 0.06

MEDIAN_LINE_C = "#1A237E"
BAND_C = "#3F51B5"
BAND_ALPHA = 0.30
COUNT_FACE = "#9E9E9E"
COUNT_EDGE = "#616161"

ZONE_COLORS = {"ideal": "#8FD694", "almost": "#F2D16B", "not": "#F5A25D"}
ZONE_ALPHA = 0.35

SYMLOG_COLUMNS = {
    "min_effective_samples": (10.0, 0.5),
    "divergent_transitions": (1.0, 1.0),
}

# (column, y-label, zone rule): ("upper", a, b), ("lower", a, b) or ("centered", c, a, b)
DIAG_PANELS = [
    ("max_r_hat",             r"$\max(\hat{R})$",        ("upper", 1.01, 1.05)),
    ("min_effective_samples", r"$\min(N_{\rm eff})$",    ("lower", 400.0, 300.0)),
    ("divergent_transitions", "Divergences",             ("upper", 0.5, 30.0)),
    (None,                    "Stars per bin",           None),
    ("mean_res_outside",      r"$\mu_{\rm res,out}$",    ("centered", 1.0, 0.02, 0.05)),
    ("std_res_outside",       r"$\sigma_{\rm res,out}$", ("centered", 1.0, 0.05, 0.10)),
    ("mean_res_inside",       r"$\mu_{\rm res,in}$",     ("centered", 1.0, 0.05, 0.10)),
    ("std_res_inside",        r"$\sigma_{\rm res,in}$",  ("centered", 1.0, 0.50, 1.00)),
    ("mean_res_full",         r"$\mu_{\rm res,full}$",   ("centered", 1.0, 0.02, 0.05)),
    ("std_res_full",          r"$\sigma_{\rm res,full}$",("centered", 1.0, 0.25, 0.50)),
]

ZONE_DESCRIPTIONS = [
    "max(R_hat): ideal <= 1.01; almost ideal 1.01-1.05; not ideal > 1.05",
    "min(N_eff): ideal >= 400; almost ideal 300-400; not ideal < 300",
    "Divergences: ideal 0 (shaded below 0.5); almost ideal 1-30; not ideal > 30",
    "min(N_eff) and divergences: symlog y-axis from 0 (linear below 10 and 1, respectively)",
    "mu_res,out and mu_res,full: ideal |1 - mu| <= 0.02; almost ideal 0.02-0.05; not ideal > 0.05",
    "mu_res,in: ideal |1 - mu| <= 0.05; almost ideal 0.05-0.10; not ideal > 0.10",
    "sigma_res,out: ideal |1 - sigma| <= 0.05; almost ideal 0.05-0.10; not ideal > 0.10",
    "sigma_res,in: ideal |1 - sigma| <= 0.50; almost ideal 0.50-1.00; not ideal > 1.00",
    "sigma_res,full: ideal |1 - sigma| <= 0.25; almost ideal 0.25-0.50; not ideal > 0.50",
]


def binned_percentiles(x, y, edges):
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    p16, p50, p84 = (np.full(n_bins, np.nan) for _ in range(3))
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = (idx == b) & np.isfinite(y)
        counts[b] = sel.sum()
        if counts[b] > 0:
            p16[b], p50[b], p84[b] = np.percentile(y[sel], [16, 50, 84])
    return p16, p50, p84, counts


def zone_spans(rule, y0, y1):
    big_lo, big_hi = min(y0, -1e30), max(y1, 1e30)
    kind = rule[0]
    if kind == "upper":
        _, a, b = rule
        return [(big_lo, a, "ideal"), (a, b, "almost"), (b, big_hi, "not")]
    if kind == "lower":
        _, a, b = rule
        return [(a, big_hi, "ideal"), (b, a, "almost"), (big_lo, b, "not")]
    _, c, a, b = rule
    return [(c - a, c + a, "ideal"),
            (c + a, c + b, "almost"), (c - b, c - a, "almost"),
            (c + b, big_hi, "not"), (big_lo, c - b, "not")]


def set_symlog_ticks(ax, linthresh, ymax):
    k0 = int(np.floor(np.log10(linthresh)))
    k1 = int(np.floor(np.log10(ymax)))
    major = [0.0] + [10.0**k for k in range(k0, k1 + 1)]
    minor = [m * 10.0**k for k in range(k0, k1 + 1) for m in range(2, 10) if m * 10.0**k <= ymax]
    ax.yaxis.set_major_locator(FixedLocator(major))
    ax.yaxis.set_minor_locator(FixedLocator(minor))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def set_x_ticks(ax, scale, xmin, xmax):
    if scale == "log":
        if xmax / xmin < 10:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def diagnostics_page(pdf, rows_sel, x, xlabel, edges, scale):
    n_bins = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:]) if scale == "log" else 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(5, 2, figsize=DIAG_FIGSIZE, sharex=True, layout="constrained")
    fig.get_layout_engine().set(wspace=DIAG_COLUMN_GAP)
    axes = axes.ravel()

    counts = np.zeros(n_bins, dtype=int)
    handles = None
    for ax, (col, ylabel, rule) in zip(axes, DIAG_PANELS):
        if col is None:
            continue
        y = column(rows_sel, col)
        p16, p50, p84, n_in_bin = binned_percentiles(x, y, edges)
        counts = np.maximum(counts, n_in_bin)

        band = ax.fill_between(centers, p16, p84, color=BAND_C, alpha=BAND_ALPHA, lw=0, zorder=2,
                               label="16th\u201384th percentile")
        (line,) = ax.plot(centers, p50, "o-", color=MEDIAN_LINE_C, lw=1.6, ms=3.2, zorder=3, label="Median")
        handles = [line, band]

        lo, hi = np.nanmin(p16), np.nanmax(p84)
        if rule[0] == "centered":
            lo, hi = min(lo, rule[1]), max(hi, rule[1])
        pad = 0.08 * (hi - lo) if hi > lo else max(0.01 * abs(hi), 0.01)
        y0, y1 = lo - pad, hi + pad
        if col in SYMLOG_COLUMNS:
            linthresh, linscale = SYMLOG_COLUMNS[col]
            ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
            y0 = 0.0
            y1 = max(hi, linthresh) * 1.8
            line.set_clip_on(False)
        for s_lo, s_hi, zone in zone_spans(rule, y0, y1):
            if s_hi > y0 and s_lo < y1:
                ax.axhspan(max(s_lo, y0), min(s_hi, y1), color=ZONE_COLORS[zone], alpha=ZONE_ALPHA, lw=0, zorder=0)
        if rule[0] == "centered":
            _, c, a, b = rule
            ax.axhline(c, color="0.15", linestyle="--", lw=1.0, zorder=2.5)
            for edge in (c - b, c - a, c + a, c + b):
                if y0 < edge < y1:
                    ax.axhline(edge, color="0.35", linestyle=":", lw=0.8, zorder=1.5)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(ylabel, fontsize=DIAG_LABEL_FONTSIZE)
        if col in SYMLOG_COLUMNS:
            set_symlog_ticks(ax, SYMLOG_COLUMNS[col][0], y1)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax_n = axes[[c for c, *_ in DIAG_PANELS].index(None)]
    ax_n.bar(centers, counts, width=np.diff(edges), color=COUNT_FACE, edgecolor=COUNT_EDGE, lw=0.6, zorder=2)
    ax_n.set_ylabel("Stars per bin", fontsize=DIAG_LABEL_FONTSIZE)
    ax_n.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    for k, ax in enumerate(axes):
        ax.set_xscale(scale)
        ax.set_xlim(edges[0], edges[-1])
        set_x_ticks(ax, scale, edges[0], edges[-1])
        ax.tick_params(axis="both", which="both", labelsize=DIAG_TICK_FONTSIZE)
        ax.grid(True, which="major", linestyle=":", alpha=0.35, zorder=1)
        ax.text(0.015, 0.94, f"({chr(97 + k)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=DIAG_PANEL_FONTSIZE, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    for ax in axes[1::2]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    for ax in axes[-2:]:
        ax.set_xlabel(xlabel, fontsize=DIAG_LABEL_FONTSIZE)
    fig.align_ylabels(axes)

    if handles:
        fig.legend(handles, [h.get_label() for h in handles], loc="outside upper center",
                   ncol=2, fontsize=DIAG_LEGEND_FONTSIZE, frameon=False)

    pdf.savefig(fig)
    plt.close(fig)
    return edges, counts


def make_diagnostics_vs_numax(rows, outdir, n_bins_log, n_bins_lin, nyq_max, nyq_desc):
    numax_map = column(rows, "nu_rmmax_map")
    numax_in = column(rows, "numax_input")
    dnu_in = column(rows, "dnu_input")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dnu_in / (0.263 * numax_in**0.772)

    ok = np.isfinite(numax_map) & (numax_map > 0) & np.isfinite(ratio) & (ratio > 0)
    n_used = int(ok.sum())
    if n_used == 0:
        print("No stars with valid numax (MAP) and input numax/dnu; diagnostics plots not made.")
        return ["Diagnostics plots: not made (no stars with valid numax MAP and input numax/dnu)."]

    kics = list(rows.keys())
    rows_sel = {kics[i]: rows[kics[i]] for i in np.flatnonzero(ok)}
    numax_map, ratio = numax_map[ok], ratio[ok]
    if numax_map.min() == numax_map.max() or ratio.min() == ratio.max():
        print("All stars share the same numax or dnu ratio; diagnostics plots not made.")
        return ["Diagnostics plots: not made (no spread in numax or in the dnu ratio)."]

    numax_label = r"$\nu_{\rm max}^{*}$ ($\mu$Hz)"
    ratio_label = r"$\Delta\nu_{\rm input} \,/\, (0.263\,\nu_{\rm max,input}^{0.772})$"

    numax_top = max(nyq_max, numax_map.max())
    if numax_map.max() > nyq_max:
        print(f"WARNING: largest numax_MAP ({numax_map.max():.2f} muHz) exceeds the Nyquist value "
              f"({nyq_max:.4f} muHz); the numax bins extend to include it.")
    edges_numax_log = np.logspace(np.log10(numax_map.min()), np.log10(numax_top), n_bins_log + 1)
    edges_numax_lin = np.linspace(0.0, numax_top, n_bins_lin + 1)
    edges_ratio_log = np.logspace(np.log10(ratio.min()), np.log10(ratio.max()), n_bins_log + 1)
    edges_ratio_lin = np.linspace(ratio.min(), ratio.max(), n_bins_lin + 1)

    pages = [
        ("numax_MAP (muHz)", numax_map, numax_label, edges_numax_log, "log",
         "lower edge = smallest numax_MAP; upper edge = largest Nyquist frequency"),
        ("numax_MAP (muHz)", numax_map, numax_label, edges_numax_lin, "linear",
         "lower edge = 0; upper edge = largest Nyquist frequency"),
        ("dnu_input / (0.263 numax_input^0.772)", ratio, ratio_label, edges_ratio_log, "log",
         "edges = smallest and largest ratio"),
        ("dnu_input / (0.263 numax_input^0.772)", ratio, ratio_label, edges_ratio_lin, "linear",
         "edges = smallest and largest ratio"),
    ]

    out_path = Path(outdir) / f"Stage1_diagnostics_vs_numax_{n_used}.pdf"
    lines = [f"[{out_path.name}]",
             f"  Stars used on every page: {n_used:,}",
             "  Panels (row by row): (a) max(R_hat), (b) min(N_eff), (c) divergences, (d) stars per bin,",
             "    (e) mu_res,out, (f) sigma_res,out, (g) mu_res,in, (h) sigma_res,in,",
             "    (i) mu_res,full, (j) sigma_res,full",
             f"  Nyquist frequency used for the numax bins: {nyq_max:.4f} muHz",
             f"    Source: {nyq_desc}",
             "  Per bin: median (line) and 16th-84th percentile range (band) of the finite values",
             "  Background zones (green = ideal, golden = almost ideal, orange = not ideal):"]
    lines += [f"    {d}" for d in ZONE_DESCRIPTIONS]

    with PdfPages(out_path) as pdf:
        for page, (xname, x, xlabel, edges, scale, edge_note) in enumerate(pages, start=1):
            edges, counts = diagnostics_page(pdf, rows_sel, x, xlabel, edges, scale)
            n_bins = len(edges) - 1
            bin_kind = "log-spaced" if scale == "log" else "equal-width (linear)"
            lines += [f"  Page {page}: x = {xname}; {scale} x-axis; {n_bins} {bin_kind} bins; "
                      f"range {edges[0]:.4f} to {edges[-1]:.4f}; {n_used:,} stars",
                      f"    ({edge_note})",
                      "    bin   lower_edge   upper_edge   n_stars"]
            lines += [f"    {b + 1:>3}   {edges[b]:>10.4f}   {edges[b + 1]:>10.4f}   {counts[b]:>7,}"
                      for b in range(n_bins)]

    print(f"\nStars used for the binned diagnostics: {n_used:,}")
    print(f"Saved diagnostics plots: {out_path.name}")
    return lines


# ============================================================
# Nyquist frequency (muHz) = 1e6 / (2 * cadence in seconds)
# ============================================================
MISSION_CADENCES = {
    "kepler": {"LC": (1765.46, "Kepler long cadence (29.4 min)"),
               "SC": (58.85, "Kepler short cadence (58.85 s)")},
    "k2":     {"LC": (1765.46, "K2 long cadence (29.4 min)"),
               "SC": (58.85, "K2 short cadence (58.85 s)")},
    "tess":   {"20s": (20.0, "TESS 20-s cadence"),
               "120s": (120.0, "TESS 2-min cadence"),
               "200s": (200.0, "TESS 200-s FFIs"),
               "600s": (600.0, "TESS 10-min FFIs"),
               "1800s": (1800.0, "TESS 30-min FFIs")},
    "plato":  {"2.5s": (2.5, "PLATO fast-camera imagettes (2.5 s)"),
               "25s": (25.0, "PLATO normal-camera imagettes (25 s)"),
               "50s": (50.0, "PLATO on-board light curves (50 s)"),
               "600s": (600.0, "PLATO on-board light curves (600 s)")},
}
DEFAULT_MISSION, DEFAULT_CADENCE = "kepler", "LC"

FREQ_UNITS_TO_MUHZ = {
    "muhz": 1.0, "uhz": 1.0, "\u00b5hz": 1.0, "\u03bchz": 1.0,
    "nhz": 1e-3, "mhz": 1e3, "hz": 1e6, "khz": 1e9,
    "c/d": 1e6 / 86400.0, "cpd": 1e6 / 86400.0, "1/d": 1e6 / 86400.0, "d-1": 1e6 / 86400.0,
}
TIME_UNITS_TO_SECONDS = {"ms": 1e-3, "s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0, "d": 86400.0}


def cadence_to_nyquist(seconds):
    return 1e6 / (2.0 * seconds)


def resolve_cadence(mission, cadence):
    m = mission.strip().lower()
    if m not in MISSION_CADENCES:
        raise ValueError(f"unknown mission {mission!r}; choose from {', '.join(MISSION_CADENCES)}")
    options = MISSION_CADENCES[m]
    key = cadence.strip()
    if key not in options and key.upper() in options:
        key = key.upper()
    if key not in options:
        raise ValueError(f"unknown cadence {cadence!r} for {m}; choose from {', '.join(options)}")
    seconds, desc = options[key]
    return cadence_to_nyquist(seconds), f"{desc}: cadence {seconds:g} s"


def parse_nyquist(text, unit=None):
    m = re.match(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$", str(text))
    if m is None:
        raise ValueError(f"could not read a number from {text!r}")
    value = float(m.group(1))
    given_unit = m.group(2) or unit
    if value <= 0:
        raise ValueError("the value must be positive")
    if not given_unit:
        return value, f"{value:g} muHz (no unit given; muHz assumed)"
    key = given_unit.strip().lower().replace(" ", "")
    if key in FREQ_UNITS_TO_MUHZ:
        nyq = value * FREQ_UNITS_TO_MUHZ[key]
        return nyq, f"{value:g} {given_unit} = {nyq:.4f} muHz"
    if key in TIME_UNITS_TO_SECONDS:
        nyq = cadence_to_nyquist(value * TIME_UNITS_TO_SECONDS[key])
        return nyq, f"{value:g} {given_unit} read as the sampling cadence; Nyquist = {nyq:.4f} muHz"
    raise ValueError(f"unknown unit {given_unit!r}")


def nyquist_from_psds(rows, psd_dir, psd_pattern):
    nyqs, missing = [], 0
    for kic in rows:
        path = Path(psd_dir) / psd_pattern.format(kic=kic)
        try:
            arr = np.load(path, mmap_mode="r")
            nyqs.append(float(arr[0, -1] if arr.shape[0] < arr.shape[1] else arr[-1, 0]))
        except (OSError, ValueError, IndexError):
            missing += 1
    if not nyqs:
        return None, "no PSDs could be read"
    desc = f"largest last frequency over {len(nyqs):,} PSDs"
    if missing:
        desc += f" ({missing:,} missing or unreadable)"
    return max(nyqs), desc


def determine_nyquist(args, rows):
    """Order: --nyquist, then --mission/--cadence, then --psd-dir, then Kepler long cadence."""
    if args.nyquist is not None:
        nyq, desc = parse_nyquist(args.nyquist, args.nyquist_unit)
        return nyq, f"given by user: {desc}"
    if args.mission or args.cadence:
        if not (args.mission and args.cadence):
            sys.exit("--mission and --cadence must be given together.")
        return resolve_cadence(args.mission, args.cadence)
    if args.psd_dir:
        nyq, desc = nyquist_from_psds(rows, args.psd_dir, args.psd_pattern)
        if nyq is not None:
            return nyq, desc
        print(f"WARNING: {desc}; using the default.")
    nyq, desc = resolve_cadence(DEFAULT_MISSION, DEFAULT_CADENCE)
    return nyq, f"default {desc} (no --nyquist, --mission/--cadence or --psd-dir given)"


def load_catalogue(path):
    catalogue = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            catalogue[int(float(r["KIC"]))] = (r["numax"].strip(), r["Delnu"].strip())
    return catalogue


def main():
    ap = argparse.ArgumentParser(description="Build the SPANDA-Stage 1 summary CSV, table and diagnostic plots.")
    ap.add_argument("--logdir", default="./logs", help="folder with the kic_<KIC>.log files (default: ./logs)")
    ap.add_argument("--catalog", default=None,
                    help="input catalogue CSV (KIC, numax, Delnu); recommended, gives full-precision inputs")
    ap.add_argument("--outdir", default=".", help="where the stage1_summary_<N> folder is created (default: .)")
    ap.add_argument("--nbins", type=int, default=17, help="log-spaced bins for the diagnostics plots")
    ap.add_argument("--nbins-linear", type=int, default=35, help="linear bins for the diagnostics plots")
    ap.add_argument("--nyquist", default=None, help="Nyquist frequency, e.g. 283.2 or '283.2 muHz'")
    ap.add_argument("--nyquist-unit", default=None, help="unit for --nyquist if not in the value")
    ap.add_argument("--mission", default=None, help="kepler, k2, tess or plato (with --cadence)")
    ap.add_argument("--cadence", default=None, help="e.g. LC/SC (Kepler, K2), 120s (TESS), 25s (PLATO)")
    ap.add_argument("--psd-dir", default=None, help="PSD folder; the largest Nyquist frequency is read from the PSDs")
    ap.add_argument("--psd-pattern", default="{kic}.npy", help="PSD file name inside --psd-dir (default: {kic}.npy)")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logs = sorted(logdir.glob("kic_*.log"))
    if not logs:
        sys.exit(f"No kic_<KIC>.log files found in {logdir}")

    catalogue = load_catalogue(args.catalog) if args.catalog else None

    rows, incomplete, failed, mismatched = {}, [], [], []
    for path in logs:
        try:
            row, status = parse_log(path, catalogue)
        except ValueError as err:
            failed.append((path.name, str(err)))
            continue
        if status == "incomplete":
            incomplete.append(path.name)
            continue
        name_kic = RE_LOGNAME.search(path.name)
        if name_kic and int(name_kic.group(1)) != int(row[0]):
            mismatched.append((path.name, row[0]))
        kic = int(row[0])
        if kic in rows:
            failed.append((path.name, f"duplicate KIC {kic}; kept the first occurrence"))
            continue
        if catalogue is not None and kic not in catalogue:
            print(f"Note: KIC {kic} not in catalogue; numax/dnu taken from its log (2 decimals).")
        rows[kic] = row

    rows = {k: rows[k] for k in sorted(rows)}
    out_dir = Path(args.outdir) / f"stage1_summary_{len(rows)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"stage1_summary_{len(rows)}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        for row in rows.values():
            writer.writerow(row)

    print(f"Log files found      : {len(logs)}")
    print(f"Completed (written)  : {len(rows)}")
    print(f"Incomplete (skipped) : {len(incomplete)}")
    print(f"Unparseable (skipped): {len(failed)}")
    for name, err in failed:
        print(f"  {name}: {err}")
    if mismatched:
        print(f"WARNING: {len(mismatched)} log(s) report a KIC different from their filename:")
        for name, kic in mismatched:
            print(f"  {name}: log reports KIC {kic}")
    print(f"Saved: {out_path}")

    if not rows:
        return

    counts, nonfinite = diagnostic_tiers(rows)
    print_diagnostics(counts, nonfinite, len(rows))
    tex_path = out_dir / f"stage1_diagnostics_table_{len(rows)}.tex"
    write_latex_table(counts, len(rows), tex_path)
    print(f"\nSaved LaTeX table: {tex_path.name}")

    details = ["SPANDA-Stage 1: details of the summary products",
               f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               f"Summary CSV: {out_path.name} ({len(rows):,} completed stars)",
               f"LaTeX table: {tex_path.name}",
               ""]
    details += make_runtime_histograms(rows, out_dir)
    details.append("")
    nyq_max, nyq_desc = determine_nyquist(args, rows)
    print(f"\nNyquist frequency used for the numax bins: {nyq_max:.4f} muHz ({nyq_desc})")
    details += make_diagnostics_vs_numax(rows, out_dir, args.nbins, args.nbins_linear, nyq_max, nyq_desc)

    details_path = out_dir / f"plot_details_{len(rows)}.txt"
    details_path.write_text("\n".join(details) + "\n")
    print(f"Saved plot details: {details_path.name}")


if __name__ == "__main__":
    main()
