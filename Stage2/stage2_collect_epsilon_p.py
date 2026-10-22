#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 2: collects the per-star <KIC>_epp_estimates.npy files written by
# stage2a_epsilon_p.py into one table, read by stage2b_radial_modes.py.
# Usage: python3 stage2_collect_epsilon_p.py [--estimates-dir DIR] [--out FILE]
# Authors: Shatanik Bhattacharya, Sapnodip Pramanik
# ============================================================
import argparse
import glob
import os

import numpy as np

COLS = ['KEPLER_ID', 'OBS_MONTHS', 'nu_rmmax_map', 'dnu_input', 'HWHM_map', 'EPSILON_P_CALC']


def main():
    ap = argparse.ArgumentParser(description="Collect the Stage 2a epsilon_p estimates into one table.")
    ap.add_argument("--estimates-dir", default="./epsilon_p_estimates", help="folder with <KIC>_epp_estimates.npy")
    ap.add_argument("--out", default="./stage2a_epsilon_p_summary.txt", help="output table")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.estimates_dir, '*_epp_estimates.npy')))
    rows = []
    for f in files:
        try:
            arr = np.load(f)
            row = [arr[c][0] for c in COLS]
        except (KeyError, ValueError):
            continue
        if all(v is not None and np.isfinite(v) for v in row):
            rows.append(row)

    rows.sort(key=lambda r: r[0])

    with open(args.out, 'w') as fh:
        fh.write('{:<12}{:>14}{:>16}{:>14}{:>14}{:>16}\n'.format(*COLS))
        for kic, obs_m, numax, dnu, hwhm, eps in rows:
            fh.write('{:<12d}{:>14.4f}{:>16.4f}{:>14.4f}{:>14.4f}{:>16.4f}\n'.format(
                int(kic), obs_m, numax, dnu, hwhm, eps))

    print(f"Wrote {len(rows)} of {len(files)} stars to {args.out}")


if __name__ == "__main__":
    main()
