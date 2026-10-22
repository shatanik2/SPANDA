#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 3a summary: builds the l=0 summary CSV from the per-star logs of
# stage3a_l0_fit.py (only completed runs), prints the sampling-diagnostic zones,
# writes them as an AASTeX table, and makes the runtime histograms and the
# diagnostics binned in the local large separation.
# Output: ./stage3a_summary_<N>/
# Usage:  python3 make_stage3a_summary.py --logdir <DIR>
# Author: Shatanik Bhattacharya
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
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter

STAT_SUFFIXES = ["median", "err_pos", "err_neg", "map", "in_1sigma"]
FIXED_COLUMNS = [
    "file_name", "kic", "delta_f", "dnu_calculated", "dnu_closest_modes_count", "freq_filter_min", "freq_filter_max",
    "num_l0_modes", "total_parameters", "gamma_guess", "de_time_sec", "de_nll", "hmc_time_sec", "max_r_hat", "min_n_eff",
    "divergent_transitions", "diag_time_sec", "gamma0_map", "profile_status",
]

NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"

RE_TOTAL   = re.compile(r"Total Pipeline Time:\s*" + NUM + r"\s*minutes")
RE_KIC     = re.compile(r"KIC\s+(\d+)\s*-\s*l=0 Mode Fitting")
RE_DELTAF  = re.compile(r"Frequency resolution delta_f\s*=\s*" + NUM)
RE_DNU     = re.compile(r"Dynamically calculated Dnu\s*=\s*" + NUM + r"\s*\S*\s*from closest\s+(\d+)\s+modes")
RE_FILTER  = re.compile(r"Filtering modes in range:\s*\[\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\]")
RE_NMODES  = re.compile(r"Identified\s+(\d+)\s+l=0 modes in range\.\s*Total parameters:\s*(\d+)")
RE_GGUESS  = re.compile(r"Computed Gamma_guess:\s*" + NUM)
RE_DE      = re.compile(r"DE Finished in\s*" + NUM + r"\s*sec\.\s*NLL:\s*" + NUM)
RE_HMC     = re.compile(r"HMC Finished in\s*" + NUM + r"\s*sec\.")
RE_RHAT    = re.compile(r"Max split R-hat \(single chain\)\s*:\s*" + NUM)
RE_NEFF    = re.compile(r"Min effective sample size\s*:\s*" + NUM)
RE_DIV     = re.compile(r"Divergent transitions\s*:\s*" + NUM, re.I)
RE_DIAG    = re.compile(r"\[TIMING\] Diagnostics calculation took\s*" + NUM + r"\s*sec")
RE_G0MAP   = re.compile(r"Gamma0 \(MAP\)\s*=\s*" + NUM + r"\s*\S*\s*->\s*(RESOLVED|UNRESOLVED)")
RE_UNC     = re.compile(r"\+\s*" + NUM + r"\s*/\s*-\s*" + NUM)
RE_LOGNAME = re.compile(r"kic_(\d+)\.log$")


def require(regex, text, what):
    m = regex.search(text)
    if m is None:
        raise ValueError(f"missing {what}")
    return m.groups()


def param_names_and_labels(K):
    prefixes = ["Gamma_0"] + [f"log_10_H_0_{i}" for i in range(K)] + [f"nu_0_{i}" for i in range(K)]
    labels = ([r"$\Gamma_0$"] + [rf"$\log_{{10}} H_{{0,{i}}}$" for i in range(K)]
              + [rf"$\nu_{{0,{i}}}$" for i in range(K)])
    return prefixes, labels


def parse_param_table(text, K, D):
    prefixes, labels = param_names_and_labels(K)
    if len(prefixes) != D:
        raise ValueError(f"total parameters ({D}) does not equal 2K+1 with K = {K}")
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "--- HMC Inferred Parameters ---")
    except StopIteration:
        raise ValueError("missing parameter table")
    rows = lines[start + 3:start + 3 + D]
    if len(rows) != D:
        raise ValueError("incomplete parameter table")

    out = {}
    for row, prefix, expected in zip(rows, prefixes, labels):
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
        for suffix, value in zip(STAT_SUFFIXES, [median, m.group(1), m.group(2), map_val, in_1sigma]):
            out[f"{prefix}_{suffix}"] = value
    return out


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    total = RE_TOTAL.search(text)
    if total is None:
        return None, "incomplete"

    (kic,) = require(RE_KIC, text, "KIC header line")
    (delta_f,) = require(RE_DELTAF, text, "frequency resolution line")
    dnu, n_closest = require(RE_DNU, text, "calculated Dnu line")
    f_min, f_max = require(RE_FILTER, text, "mode filter range line")
    K, D = require(RE_NMODES, text, "number of modes line")
    (gamma_guess,) = require(RE_GGUESS, text, "Gamma_guess line")
    de_time, de_nll = require(RE_DE, text, "DE timing line")
    (hmc_time,) = require(RE_HMC, text, "HMC timing line")
    (r_hat,) = require(RE_RHAT, text, "split R-hat line")
    (n_eff,) = require(RE_NEFF, text, "effective sample size line")
    (div,) = require(RE_DIV, text, "divergences line")
    (diag_time,) = require(RE_DIAG, text, "diagnostics timing line")
    g0_map, status = require(RE_G0MAP, text, "Gamma0 (MAP) line")

    row = {
        "file_name": path.name, "kic": kic, "delta_f": delta_f,
        "dnu_calculated": dnu, "dnu_closest_modes_count": n_closest,
        "freq_filter_min": f_min, "freq_filter_max": f_max,
        "num_l0_modes": K, "total_parameters": D, "gamma_guess": gamma_guess,
        "de_time_sec": de_time, "de_nll": de_nll, "hmc_time_sec": hmc_time,
        "max_r_hat": r_hat, "min_n_eff": n_eff, "divergent_transitions": div,
        "diag_time_sec": diag_time, "gamma0_map": g0_map, "profile_status": status,
        "total_pipeline_time_min": total.group(1),
    }
    row.update(parse_param_table(text, int(K), int(D)))
    return row, "ok"


def build_header(rows):
    k_max = max((int(r["num_l0_modes"]) for r in rows.values()), default=0)
    header = list(FIXED_COLUMNS)
    header += [f"Gamma_0_{s}" for s in STAT_SUFFIXES]
    for prefix in ("log_10_H_0", "nu_0"):
        for i in range(k_max):
            header += [f"{prefix}_{i}_{s}" for s in STAT_SUFFIXES]
    header += ["total_pipeline_time_min"]
    return header


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def diagnostic_tiers(rows):
    counts = {k: 0 for k in ["rhat_1", "rhat_2", "rhat_3", "neff_1", "neff_2", "neff_3", "div_1", "div_2", "div_3"]}
    for row in rows.values():
        r = to_float(row["max_r_hat"])
        counts["rhat_1" if r <= 1.01 else "rhat_2" if r <= 1.05 else "rhat_3"] += 1
        n = to_float(row["min_n_eff"])
        counts["neff_1" if n >= 400 else "neff_2" if n >= 300 else "neff_3"] += 1
        d = to_float(row["divergent_transitions"])
        counts["div_1" if d == 0 else "div_2" if d <= 30 else "div_3"] += 1
    return counts


def print_diagnostics(counts, total):
    groups = [
        ("max(split R_hat), single chain", [("<= 1.01", "rhat_1"), ("1.01 < max(R_hat) <= 1.05", "rhat_2"), ("> 1.05 (or non-finite)", "rhat_3")]),
        ("min(N_eff)", [(">= 400", "neff_1"), ("300 <= min(N_eff) < 400", "neff_2"), ("< 300 (or non-finite)", "neff_3")]),
        ("Divergent Transitions", [("= 0", "div_1"), ("1 <= Divergences <= 30", "div_2"), ("> 30 (or non-finite)", "div_3")]),
    ]
    print("\n--- HMC Sampling Diagnostics: Zone Breakdown ---")
    print(f"Total stars: N = {total:,}")
    for name, tiers in groups:
        print(f"\n{name}")
        for label, key in tiers:
            print(f"  {label:<40} {counts[key]:>8,}  {pct(counts[key], total):6.2f} %")


def write_latex_table(counts, total, out_path, label, title):
    def line(key):
        return f"{counts[key]:,} & {pct(counts[key], total):.2f}"

    n_str = f"{total:,}".replace(",", "{,}")
    tex = rf"""\begin{{deluxetable*}}{{llcc}}
\tablecaption{{{title}\label{{{label}}}}}
\tablehead{{\colhead{{Parameter}} & \colhead{{Zone}} & \colhead{{Star Count}} & \colhead{{Percentage (\%)}}}}
\startdata
$\max(\hat{{R}}_{{\rm split}})$ & $\leq 1.01$ & {line("rhat_1")} \\
 & $1.01 < \max(\hat{{R}}_{{\rm split}}) \leq 1.05$ & {line("rhat_2")} \\
 & $> 1.05$ & {line("rhat_3")} \\
\hline
$\min(N_{{\rm eff}})$ & $\geq 400$ & {line("neff_1")} \\
 & $300 \leq \min(N_{{\rm eff}}) < 400$ & {line("neff_2")} \\
 & $< 300$ & {line("neff_3")} \\
\hline
Divergent Transitions & $= 0$ & {line("div_1")} \\
 & $1 \leq$ Divergences $\leq 30$ & {line("div_2")} \\
 & $> 30$ & {line("div_3")} \\
\enddata
\tablecomments{{Total stars: $N = {n_str}$. Each star was sampled with a single HMC chain; $\hat{{R}}_{{\rm split}}$ compares the two halves of that chain, and the maximum is taken over all fitted parameters. $\min(N_{{\rm eff}})$ is the smallest effective sample size over all fitted parameters.}}
\end{{deluxetable*}}
"""
    Path(out_path).write_text(tex)


BAR_FACE, BAR_EDGE = "#A7A6EA", "#8786DD"
MEAN_C, MEDIAN_C, MINUS1S_C, PLUS1S_C = "#008080", "#E06653", "#000000", "#B0307C"


def plot_hist(pdf, data, xlabel):
    mean = np.mean(data)
    p16, p50, p84 = np.percentile(data, [15.87, 50.0, 84.13])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(data, bins=60, color=BAR_FACE, edgecolor=BAR_EDGE, linewidth=0.6)
    ax.axvline(mean, color=MEAN_C, linestyle="--", lw=2.5, label=f"Mean: {mean:.2f} s")
    ax.axvline(p50, color=MEDIAN_C, linestyle="--", lw=2.5, label=f"Median (50.00%): {p50:.2f} s")
    ax.axvline(p16, color=MINUS1S_C, linestyle=":", lw=2.5, label=rf"$-1\sigma$ (15.87%): {p16:.2f} s")
    ax.axvline(p84, color=PLUS1S_C, linestyle="-.", lw=2.5, label=rf"$+1\sigma$ (84.13%): {p84:.2f} s")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Count (Number of Stars)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=14, edgecolor="0.3")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return mean, p16, p50, p84


def column(rows, name):
    return np.array([to_float(r.get(name)) for r in rows.values()], dtype=float)


def make_runtime_histograms(rows, outdir, stage):
    de, hmc = column(rows, "de_time_sec"), column(rows, "hmc_time_sec")
    ok = np.isfinite(de) & np.isfinite(hmc)
    de, hmc = de[ok], hmc[ok]
    if len(de) == 0:
        return ["Runtime histograms: not made (no valid DE/HMC times)."]
    out_path = Path(outdir) / f"{stage}_DE_HMC_times_{len(de)}.pdf"
    with PdfPages(out_path) as pdf:
        s_de = plot_hist(pdf, de, "DE Time (seconds)")
        s_hmc = plot_hist(pdf, hmc, "HMC Time (seconds)")
    print(f"Saved runtime histograms: {out_path.name}")
    lines = [f"[{out_path.name}]", f"  Stars used: {len(de):,} (finite DE and HMC times)"]
    for name, (mean, p16, p50, p84) in (("DE time (s)", s_de), ("HMC time (s)", s_hmc)):
        lines.append(f"  {name}: mean = {mean:.2f}, median = {p50:.2f}, 15.87th = {p16:.2f}, 84.13th = {p84:.2f}")
    return lines


ZONE_COLORS = {"z1": "#8FD694", "z2": "#F2D16B", "z3": "#F5A25D"}
SYMLOG_COLUMNS = {"min_n_eff": (10.0, 0.5), "divergent_transitions": (1.0, 1.0)}
DIAG_PANELS = [
    ("max_r_hat",             r"$\max(\hat{R}_{\rm split})$", ("upper", 1.01, 1.05)),
    ("min_n_eff",             r"$\min(N_{\rm eff})$",          ("lower", 400.0, 300.0)),
    ("divergent_transitions", "Divergences",                   ("upper", 0.5, 30.0)),
    (None,                    "Stars per bin",                 None),
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
    kind, a, b = rule
    if kind == "upper":
        return [(big_lo, a, "z1"), (a, b, "z2"), (b, big_hi, "z3")]
    return [(a, big_hi, "z1"), (b, a, "z2"), (big_lo, b, "z3")]


def set_symlog_ticks(ax, linthresh, ymax):
    k0, k1 = int(np.floor(np.log10(linthresh))), int(np.floor(np.log10(ymax)))
    ax.yaxis.set_major_locator(FixedLocator([0.0] + [10.0**k for k in range(k0, k1 + 1)]))
    ax.yaxis.set_minor_locator(FixedLocator([m * 10.0**k for k in range(k0, k1 + 1) for m in range(2, 10) if m * 10.0**k <= ymax]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def diagnostics_page(pdf, rows_sel, x, xlabel, edges, scale):
    n_bins = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:]) if scale == "log" else 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    axes = axes.ravel()
    counts = np.zeros(n_bins, dtype=int)
    handles = None
    for ax, (col, ylabel, rule) in zip(axes, DIAG_PANELS):
        if col is None:
            continue
        y = column(rows_sel, col)
        p16, p50, p84, n_in_bin = binned_percentiles(x, y, edges)
        counts = np.maximum(counts, n_in_bin)
        band = ax.fill_between(centers, p16, p84, color="#3F51B5", alpha=0.30, lw=0, zorder=2, label="16th\u201384th percentile")
        (line,) = ax.plot(centers, p50, "o-", color="#1A237E", lw=1.6, ms=3.2, zorder=3, label="Median")
        handles = [line, band]
        lo, hi = np.nanmin(p16), np.nanmax(p84)
        pad = 0.08 * (hi - lo) if hi > lo else max(0.01 * abs(hi), 0.01)
        y0, y1 = lo - pad, hi + pad
        if col in SYMLOG_COLUMNS:
            linthresh, linscale = SYMLOG_COLUMNS[col]
            ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
            y0, y1 = 0.0, max(hi, linthresh) * 1.8
            line.set_clip_on(False)
        for s_lo, s_hi, zone in zone_spans(rule, y0, y1):
            if s_hi > y0 and s_lo < y1:
                ax.axhspan(max(s_lo, y0), min(s_hi, y1), color=ZONE_COLORS[zone], alpha=0.35, lw=0, zorder=0)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(ylabel, fontsize=10.5)
        if col in SYMLOG_COLUMNS:
            set_symlog_ticks(ax, SYMLOG_COLUMNS[col][0], y1)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_n = axes[3]
    ax_n.bar(centers, counts, width=np.diff(edges), color="#9E9E9E", edgecolor="#616161", lw=0.6, zorder=2)
    ax_n.set_ylabel("Stars per bin", fontsize=10.5)
    ax_n.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    for k, ax in enumerate(axes):
        ax.set_xscale(scale)
        ax.set_xlim(edges[0], edges[-1])
        if scale == "log":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5) if edges[-1] / edges[0] < 10 else LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="both", which="both", labelsize=9)
        ax.grid(True, which="major", linestyle=":", alpha=0.35, zorder=1)
        ax.text(0.015, 0.94, f"({chr(97 + k)})", transform=ax.transAxes, ha="left", va="top", fontsize=9, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    for ax in axes[1::2]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    for ax in axes[-2:]:
        ax.set_xlabel(xlabel, fontsize=10.5)
    fig.align_ylabels(axes)
    if handles:
        fig.legend(handles, [h.get_label() for h in handles], loc="outside upper center", ncol=2, fontsize=9.5, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)
    return counts


def make_diagnostics_vs_dnu(rows, outdir, stage, n_bins_log, n_bins_lin):
    dnu = column(rows, "dnu_calculated")
    ok = np.isfinite(dnu) & (dnu > 0)
    if ok.sum() == 0 or dnu[ok].min() == dnu[ok].max():
        return ["Diagnostics plots: not made (no spread in dnu_calculated)."]
    kics = list(rows.keys())
    rows_sel = {kics[i]: rows[kics[i]] for i in np.flatnonzero(ok)}
    dnu = dnu[ok]
    out_path = Path(outdir) / f"{stage}_diagnostics_vs_dnu_{len(dnu)}.pdf"
    with PdfPages(out_path) as pdf:
        for edges, scale in ((np.logspace(np.log10(dnu.min()), np.log10(dnu.max()), n_bins_log + 1), "log"),
                             (np.linspace(dnu.min(), dnu.max(), n_bins_lin + 1), "linear")):
            diagnostics_page(pdf, rows_sel, dnu, r"$\Delta\nu_{\rm calc}$ ($\mu$Hz)", edges, scale)
    print(f"Saved diagnostics plots: {out_path.name}")
    return [f"[{out_path.name}]", f"  Stars used: {len(dnu):,}; x = dnu_calculated; pages: log and linear binning"]


def main():
    ap = argparse.ArgumentParser(description="Build the SPANDA-Stage 3a (l=0) summary CSV, table and plots.")
    ap.add_argument("--logdir", default="./logs_l0", help="folder with the kic_<KIC>.log files (default: ./logs_l0)")
    ap.add_argument("--outdir", default=".", help="where the stage3a_summary_<N> folder is created (default: .)")
    ap.add_argument("--nbins", type=int, default=17, help="log-spaced bins for the diagnostics plots")
    ap.add_argument("--nbins-linear", type=int, default=35, help="linear bins for the diagnostics plots")
    args = ap.parse_args()

    logs = sorted(Path(args.logdir).glob("kic_*.log"))
    if not logs:
        sys.exit(f"No kic_<KIC>.log files found in {args.logdir}")

    rows, incomplete, failed, mismatched = {}, [], [], []
    for path in logs:
        try:
            row, status = parse_log(path)
        except ValueError as err:
            failed.append((path.name, str(err)))
            continue
        if status == "incomplete":
            incomplete.append(path.name)
            continue
        name_kic = RE_LOGNAME.search(path.name)
        if name_kic and int(name_kic.group(1)) != int(row["kic"]):
            mismatched.append((path.name, row["kic"]))
        kic = int(row["kic"])
        if kic in rows:
            failed.append((path.name, f"duplicate KIC {kic}; kept the first occurrence"))
            continue
        rows[kic] = row

    rows = {k: rows[k] for k in sorted(rows)}
    header = build_header(rows)
    out_dir = Path(args.outdir) / f"stage3a_summary_{len(rows)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage3a_summary_{len(rows)}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="", extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)

    print(f"Log files found      : {len(logs)}")
    print(f"Completed (written)  : {len(rows)}")
    print(f"Incomplete (skipped) : {len(incomplete)}")
    print(f"Unparseable (skipped): {len(failed)}")
    for name, err in failed:
        print(f"  {name}: {err}")
    for name, kic in mismatched:
        print(f"WARNING: {name} reports KIC {kic}")
    print(f"Saved: {out_path}")
    if not rows:
        return

    counts = diagnostic_tiers(rows)
    print_diagnostics(counts, len(rows))
    n_res = sum(r["profile_status"] == "RESOLVED" for r in rows.values())
    print(f"\nProfile status of the MAP Gamma_0: RESOLVED = {n_res:,}, UNRESOLVED = {len(rows) - n_res:,}")
    tex_path = out_dir / f"stage3a_diagnostics_table_{len(rows)}.tex"
    write_latex_table(counts, len(rows), tex_path, "tab:spanda_stage3a_diagnostics",
                      r"HMC Sampling Diagnostics for the $\ell = 0$ Mode Fits of \texttt{SPANDA-Stage 3a}")
    print(f"Saved LaTeX table: {tex_path.name}")

    details = ["SPANDA-Stage 3a: details of the summary products",
               f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               f"Summary CSV: {out_path.name} ({len(rows):,} completed stars; {len(header)} columns)",
               f"Profile status of the MAP Gamma_0: RESOLVED = {n_res:,}, UNRESOLVED = {len(rows) - n_res:,}", ""]
    details += make_runtime_histograms(rows, out_dir, "Stage3a") + [""]
    details += make_diagnostics_vs_dnu(rows, out_dir, "Stage3a", args.nbins, args.nbins_linear)
    (out_dir / f"plot_details_{len(rows)}.txt").write_text("\n".join(details) + "\n")


if __name__ == "__main__":
    main()
