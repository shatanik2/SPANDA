#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 3b summary: builds the l=2 summary CSV from the per-star logs of
# stage3b_l2_fit.py (only completed runs), prints the sampling-diagnostic zones,
# writes them as an AASTeX table, and makes the runtime histograms and the
# diagnostics binned in the local large separation.
# Output: ./stage3b_summary_<N>/
# Usage:  python3 make_stage3b_summary.py --logdir <DIR>
# Author: Shatanik Bhattacharya
# ============================================================
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

from make_stage3a_summary import (STAT_SUFFIXES, NUM, RE_UNC, RE_LOGNAME, require, diagnostic_tiers,
                                  print_diagnostics, write_latex_table, make_runtime_histograms,
                                  make_diagnostics_vs_dnu)

FIXED_COLUMNS = [
    "file_name", "kic", "delta_f", "dnu_calculated", "d02_guess", "num_l2_modes", "total_parameters",
    "gamma0_fixed", "profile_status", "de_time_sec", "de_nll", "hmc_time_sec", "max_r_hat", "min_n_eff",
    "divergent_transitions", "diag_time_sec",
]

RE_TOTAL   = re.compile(r"Total Pipeline Time:\s*" + NUM + r"\s*minutes")
RE_KIC     = re.compile(r"KIC\s+(\d+)\s*-\s*l=2 Mode Fitting")
RE_DELTAF  = re.compile(r"Frequency resolution delta_f\s*=\s*" + NUM)
RE_DNU     = re.compile(r"Dnu_calculated \(from the l=0 fit\):\s*" + NUM)
RE_NMODES  = re.compile(r"Loaded\s+(\d+)\s+fixed l=0 modes")
RE_G0_D02  = re.compile(r"Fixed Gamma0:\s*" + NUM + r"\s*\S*,\s*Computed d02_guess:\s*" + NUM)
RE_STATUS  = re.compile(r"Fixed Gamma0 status ->\s*(RESOLVED|UNRESOLVED)")
RE_DE      = re.compile(r"DE Finished in\s*" + NUM + r"\s*sec\.\s*NLL:\s*" + NUM)
RE_HMC     = re.compile(r"HMC Finished in\s*" + NUM + r"\s*sec\.")
RE_RHAT    = re.compile(r"Max split R-hat \(single chain\)\s*:\s*" + NUM)
RE_NEFF    = re.compile(r"Min effective sample size\s*:\s*" + NUM)
RE_DIV     = re.compile(r"Divergent transitions\s*:\s*" + NUM, re.I)
RE_DIAG    = re.compile(r"\[TIMING\] Diagnostics calculation took\s*" + NUM + r"\s*sec")


def param_names_and_labels(K):
    prefixes = ["inc_2", "nu_rot_2", "V2"] + [f"nu_2_{i}" for i in range(K)]
    labels = [r"$i_{2}$ (deg)", r"$\Upsilon_{2}$", r"$V_2$"] + [rf"$\nu_{{2,{i}}}$" for i in range(K)]
    return prefixes, labels


def parse_param_table(text, K):
    prefixes, labels = param_names_and_labels(K)
    D = len(prefixes)
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
    (dnu,) = require(RE_DNU, text, "Dnu_calculated line")
    (K,) = require(RE_NMODES, text, "number of modes line")
    gamma0, d02 = require(RE_G0_D02, text, "fixed Gamma0 / d02 line")
    (status,) = require(RE_STATUS, text, "Gamma0 status line")
    de_time, de_nll = require(RE_DE, text, "DE timing line")
    (hmc_time,) = require(RE_HMC, text, "HMC timing line")
    (r_hat,) = require(RE_RHAT, text, "split R-hat line")
    (n_eff,) = require(RE_NEFF, text, "effective sample size line")
    (div,) = require(RE_DIV, text, "divergences line")
    (diag_time,) = require(RE_DIAG, text, "diagnostics timing line")

    row = {
        "file_name": path.name, "kic": kic, "delta_f": delta_f, "dnu_calculated": dnu, "d02_guess": d02,
        "num_l2_modes": K, "total_parameters": str(int(K) + 3), "gamma0_fixed": gamma0, "profile_status": status,
        "de_time_sec": de_time, "de_nll": de_nll, "hmc_time_sec": hmc_time,
        "max_r_hat": r_hat, "min_n_eff": n_eff, "divergent_transitions": div,
        "diag_time_sec": diag_time, "total_pipeline_time_min": total.group(1),
    }
    row.update(parse_param_table(text, int(K)))
    return row, "ok"


def build_header(rows):
    k_max = max((int(r["num_l2_modes"]) for r in rows.values()), default=0)
    header = list(FIXED_COLUMNS)
    for prefix in ["inc_2", "nu_rot_2", "V2"] + [f"nu_2_{i}" for i in range(k_max)]:
        header += [f"{prefix}_{s}" for s in STAT_SUFFIXES]
    header += ["total_pipeline_time_min"]
    return header


def main():
    ap = argparse.ArgumentParser(description="Build the SPANDA-Stage 3b (l=2) summary CSV, table and plots.")
    ap.add_argument("--logdir", default="./logs_l2", help="folder with the kic_<KIC>.log files (default: ./logs_l2)")
    ap.add_argument("--outdir", default=".", help="where the stage3b_summary_<N> folder is created (default: .)")
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
    out_dir = Path(args.outdir) / f"stage3b_summary_{len(rows)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage3b_summary_{len(rows)}.csv"
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
    tex_path = out_dir / f"stage3b_diagnostics_table_{len(rows)}.tex"
    write_latex_table(counts, len(rows), tex_path, "tab:spanda_stage3b_diagnostics",
                      r"HMC Sampling Diagnostics for the $\ell = 2$ Mode Fits of \texttt{SPANDA-Stage 3b}")
    print(f"Saved LaTeX table: {tex_path.name}")

    details = ["SPANDA-Stage 3b: details of the summary products",
               f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               f"Summary CSV: {out_path.name} ({len(rows):,} completed stars; {len(header)} columns)", ""]
    details += make_runtime_histograms(rows, out_dir, "Stage3b") + [""]
    details += make_diagnostics_vs_dnu(rows, out_dir, "Stage3b", args.nbins, args.nbins_linear)
    (out_dir / f"plot_details_{len(rows)}.txt").write_text("\n".join(details) + "\n")


if __name__ == "__main__":
    main()
