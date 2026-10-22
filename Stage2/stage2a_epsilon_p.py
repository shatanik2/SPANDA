#!/usr/bin/env python3
"""
SPANDA-Stage 2a: estimate of the p-mode phase offset epsilon_p for one star,
from a collapsed-echelle matched filter weighted by the epsilon-dnu prior of
Wang et al. (2026).

Usage:
    python3 stage2a_epsilon_p.py <KIC> --stage1-csv <CSV> --psd-dir <DIR> [--plot] [--no-file]

Authors: Sapnodip Pramanik, Shatanik Bhattacharya
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIC_COL = 'kic'
NUMAX_COL = 'nu_rmmax_map'
DNU_COL = 'dnu_input'
HWHM_COL = 'HWHM_map'

BKG_COLS = [
    "log_10_a_1_over_b_1_map", "log_10_b_1_map", "log_10_a_2_map",
    "log_10_a_3_map", "log_10_b_2_map", "log_10_b_3_map", "log_10_W_map"
]

N_BINS = 100
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
XCENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
EPS_GRID = np.linspace(0.0, 1.0, 2000, endpoint=False)

DELTA_FLOOR = 0.02

# epsilon-dnu relation (Wang et al. 2026) and prior width parameters
EPS_REL_A = 0.610
EPS_REL_B = 0.625
EPS_PRIOR_FWHM = 0.50
EPS_PRIOR_FWHM_EDGE_BOOST = 0.60

MIN_POINTS_PER_ORDER = 8


def _circular_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


DIST0 = _circular_dist(XCENTERS[None, :], EPS_GRID[:, None])
DIST0_SQ = DIST0 ** 2


# ==========================================
# Matched filter and prior
# ==========================================
def build_filter_matrix(delta: float) -> np.ndarray:
    """(n_eps, n_bins) matched-filter weights for the l=0 and l=2 ridges."""
    delta = max(delta, DELTA_FLOOR)
    sigma = (delta / 6.0) / np.sqrt(2.0 * np.log(2.0))
    inv_2sigma2 = -0.5 / (sigma * sigma)

    eps_shifted = (EPS_GRID - delta) % 1.0
    dist2 = _circular_dist(XCENTERS[None, :], eps_shifted[:, None])

    filt = np.exp(DIST0_SQ * inv_2sigma2) + np.exp(dist2 ** 2 * inv_2sigma2)
    filt /= filt.sum(axis=1, keepdims=True)
    return filt


def get_filter_at_epsilon(epsilon: float, delta: float):
    delta = max(delta, DELTA_FLOOR)
    sigma = (delta / 6.0) / np.sqrt(2.0 * np.log(2.0))
    inv_2sigma2 = -0.5 / (sigma * sigma)

    dist_l0 = _circular_dist(XCENTERS, epsilon)
    eps_l2 = (epsilon - delta) % 1.0
    dist_l2 = _circular_dist(XCENTERS, eps_l2)

    filter_l0 = np.exp(dist_l0 ** 2 * inv_2sigma2)
    filter_l2 = np.exp(dist_l2 ** 2 * inv_2sigma2)
    filt = filter_l0 + filter_l2
    filt /= filt.sum()
    return filt, filter_l0, filter_l2


def fold_power(freq: np.ndarray, power: np.ndarray, dnu: float):
    """Mean power per phase bin."""
    x = (freq / dnu) % 1.0
    sum_p, _ = np.histogram(x, bins=BIN_EDGES, weights=power)
    counts, _ = np.histogram(x, bins=BIN_EDGES)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_p = np.divide(sum_p, counts, out=np.zeros_like(sum_p), where=counts > 0)
    return mean_p, counts


def normalize_collapsed(collapsed: np.ndarray) -> np.ndarray:
    out = collapsed.astype(np.float64).copy()
    max_p = out.max()
    if max_p > 0:
        out = out / max_p
    std_p = out.std()
    if std_p > 0:
        out = (out - out.mean()) / std_p
    else:
        out = out - out.mean()
    return out


def find_local_maxima(scores: np.ndarray) -> np.ndarray:
    s_prev = np.roll(scores, 1)
    s_next = np.roll(scores, -1)
    is_peak = (scores > s_prev) & (scores > s_next)
    idx = np.nonzero(is_peak)[0]
    order = np.argsort(scores[idx])[::-1]
    return idx[order]


def epsilon_prior_weight(eps_grid: np.ndarray, dnu: float):
    """Circular super-Gaussian (order 6) prior around the epsilon-dnu relation;
    the width parameter grows as the predicted epsilon approaches 0 or 1."""
    eps_pred = (EPS_REL_A + EPS_REL_B * np.log10(dnu)) % 1.0
    edge_dist = min(eps_pred, 1.0 - eps_pred)
    fwhm = EPS_PRIOR_FWHM + EPS_PRIOR_FWHM_EDGE_BOOST * (1 - (edge_dist) / 0.5) ** 10
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    d = _circular_dist(eps_grid, eps_pred)
    prior = np.exp(-0.5 * (d / sigma) ** 6)
    return prior, eps_pred, sigma, fwhm


def circular_std(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return np.nan
    theta = 2 * np.pi * v
    C, S = np.mean(np.cos(theta)), np.mean(np.sin(theta))
    R = min(np.hypot(C, S), 1.0)
    return np.sqrt(-2.0 * np.log(R)) / (2 * np.pi)


def circular_span(center: float, halfwidth: float):
    lo, hi = center - halfwidth, center + halfwidth
    if lo < 0:
        return [(0.0, hi), (1.0 + lo, 1.0)]
    if hi > 1:
        return [(lo, 1.0), (0.0, hi - 1.0)]
    return [(lo, hi)]


# ==========================================
# Echelle grid (diagnostics only)
# ==========================================
def build_echelle(freq, power, numax, dnu, hwhm, min_points=MIN_POINTS_PER_ORDER):
    lower, upper = numax - 3.0 * hwhm, numax + 3.0 * hwhm
    n_orders = max(1, int(np.ceil((upper - lower) / dnu)))
    order_edges = lower + dnu * np.arange(n_orders + 1)
    order_edges[-1] = max(order_edges[-1], upper)

    in_range = (freq >= lower) & (freq <= upper)
    f_in, p_in = freq[in_range], power[in_range]

    order_idx = np.clip(np.searchsorted(order_edges, f_in, side='right') - 1, 0, n_orders - 1)
    phase = (f_in / dnu) % 1.0

    order_bin_edges = np.arange(n_orders + 1) - 0.5
    sum_grid, _, _ = np.histogram2d(order_idx, phase, bins=[order_bin_edges, BIN_EDGES], weights=p_in)
    cnt_grid, _, _ = np.histogram2d(order_idx, phase, bins=[order_bin_edges, BIN_EDGES])

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_grid = np.divide(sum_grid, cnt_grid, out=np.zeros_like(sum_grid), where=cnt_grid > 0)

    valid_rows = cnt_grid.sum(axis=1) >= min_points
    order_centers = 0.5 * (order_edges[:-1] + order_edges[1:])
    return mean_grid, valid_rows, order_centers, order_edges


def order_by_order_consistency(mean_grid, valid_rows, delta):
    row_max = mean_grid.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    norm = mean_grid / row_max
    row_mean = norm.mean(axis=1, keepdims=True)
    row_std = norm.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    norm = (norm - row_mean) / row_std

    filt = build_filter_matrix(delta)
    scores_matrix = filt @ norm.T
    best_idx = np.argmax(scores_matrix, axis=0)
    return np.where(valid_rows, EPS_GRID[best_idx], np.nan)


# ==========================================
# Input / output
# ==========================================
def load_star_row(kic: int, csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for c in (KIC_COL, NUMAX_COL, DNU_COL, HWHM_COL):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    row = df.loc[df[KIC_COL] == kic]
    if row.empty:
        raise ValueError(f"KIC {kic} not found in {csv_path}")
    row = row.iloc[0]
    numax, dnu, hwhm = float(row[NUMAX_COL]), float(row[DNU_COL]), float(row[HWHM_COL])
    if not (numax > 0 and dnu > 0 and hwhm > 0):
        raise ValueError(f"KIC {kic} has invalid numax/dnu/HWHM in {csv_path}")
    return numax, dnu, hwhm


def load_bkg_params(kic: int, csv_path: str) -> np.ndarray:
    df_bkg = pd.read_csv(csv_path, low_memory=False)
    df_bkg.columns = df_bkg.columns.str.strip()
    bkg_row = df_bkg[df_bkg[KIC_COL] == kic]
    if bkg_row.empty:
        raise ValueError(f"KIC {kic} not found in {csv_path}")
    return bkg_row[BKG_COLS].values[0].astype(np.float64)


def eval_bkg_model(x, bkg_params, nyq):
    """Granulation + white-noise background (no Gaussian envelope)."""
    x = np.asarray(x, dtype=np.float64)
    log10_a1_b1, log10_b1, log10_a2, log10_a3, log10_b2, log10_b3, log10_W = bkg_params

    a1_b1 = 10.0 ** log10_a1_b1
    b1 = 10.0 ** log10_b1
    a2, a3 = 10.0 ** log10_a2, 10.0 ** log10_a3
    b2, b3 = 10.0 ** log10_b2, 10.0 ** log10_b3
    W = 10.0 ** log10_W

    t1 = a1_b1 / (1.0 + (x / b1) ** 2)
    t2 = (a2 / b2) / (1.0 + (x / b2) ** 4)
    t3 = (a3 / b3) / (1.0 + (x / b3) ** 4)
    sinc_sq = np.sinc(0.5 * x / nyq) ** 2

    return t1 + sinc_sq * (t2 + t3) + W


def bkg_normalized_power(freq, power, bkg_params, nyq):
    bkg = eval_bkg_model(freq, bkg_params, nyq)
    ratio = np.clip(power, 1e-30, None) / np.clip(bkg, 1e-30, None)
    lo, hi = ratio.min(), ratio.max()
    if hi > lo:
        return (ratio - lo) / (hi - lo)
    return np.zeros_like(ratio)


def bkg_normalized_log_power(freq, power, bkg_params, nyq):
    bkg = eval_bkg_model(freq, bkg_params, nyq)
    ratio = np.clip(power, 1e-30, None) / np.clip(bkg, 1e-30, None)
    log_ratio = np.log10(ratio)
    lo, hi = log_ratio.min(), log_ratio.max()
    if hi > lo:
        return (log_ratio - lo) / (hi - lo)
    return np.zeros_like(log_ratio)


def load_psd(psd_dir, pattern, kic):
    path = os.path.join(psd_dir, pattern.format(kic=kic))
    if not os.path.exists(path):
        raise FileNotFoundError(f"PSD file not found: {path}")
    arr = np.load(path, mmap_mode='r')
    if arr.shape[0] < arr.shape[1]:
        return np.asarray(arr[0]), np.asarray(arr[1])
    return np.asarray(arr[:, 0]), np.asarray(arr[:, 1])


def save_estimates(result: dict, diag: dict, outdir: str) -> str:
    """One-row structured array <KIC>_epp_estimates.npy."""
    os.makedirs(outdir, exist_ok=True)

    freq = diag['freq']
    delta_f = float(np.median(np.diff(freq)))
    obs_seconds = 1.0e6 / delta_f
    seconds_per_month = 86400.0 * 365.25 / 12.0
    obs_months = obs_seconds / seconds_per_month

    dtype = [
        ('KEPLER_ID', 'i8'),
        ('OBS_MONTHS', 'f8'),
        ('nu_rmmax_map', 'f8'),
        ('dnu_input', 'f8'),
        ('HWHM_map', 'f8'),
        ('EPSILON_P_CALC', 'f8'),
    ]
    arr = np.array(
        [(result['kic'], obs_months, result['numax'], result['dnu'], result['hwhm'], result['epsilon_p'])],
        dtype=dtype,
    )
    path = os.path.join(outdir, f"{result['kic']}_epp_estimates.npy")
    np.save(path, arr)
    return path


# ==========================================
# Estimation
# ==========================================
def estimate_epsilon_p(kic: int, csv_path: str, psd_dir: str, psd_pattern: str):
    numax, dnu, hwhm = load_star_row(kic, csv_path)
    freq, power = load_psd(psd_dir, psd_pattern, kic)

    w = max(hwhm, dnu)
    lower, upper = numax - 3.0 * w, numax + 3.0 * w
    mask = (freq >= lower) & (freq <= upper)
    if not np.any(mask):
        raise RuntimeError(f"KIC {kic}: no frequency points inside numax +/- 3*max(HWHM, dnu) window")

    f_masked, p_masked = freq[mask], power[mask]

    bkg_params = load_bkg_params(kic, csv_path)
    nyq = float(freq[-1])
    norm_power = bkg_normalized_power(f_masked, p_masked, bkg_params, nyq)

    collapsed_raw, counts = fold_power(f_masked, norm_power, dnu)
    collapsed = normalize_collapsed(collapsed_raw)

    delta = ((0.121 * dnu + 0.047) / dnu) % 1.0
    filt = build_filter_matrix(delta)

    scores_raw = filt @ collapsed
    scores = scores_raw - scores_raw.min()
    score_max = scores.max()
    if score_max > 0:
        scores /= score_max

    prior_curve, eps_pred, prior_sigma, prior_fwhm = epsilon_prior_weight(EPS_GRID, dnu)
    combined_score = scores * prior_curve

    best_idx = int(np.argmax(combined_score))
    chosen_eps = float(EPS_GRID[best_idx])

    epsilon_p_scaled = chosen_eps % 1.0
    epsilon_p_edge_distance = min(epsilon_p_scaled, 1.0 - epsilon_p_scaled)

    best_filter_score = float(scores[best_idx])
    best_prior_weight = float(prior_curve[best_idx])
    best_combined_score = float(combined_score[best_idx])

    raw_filter_idx = int(np.argmax(scores))
    top_eps = float(EPS_GRID[raw_filter_idx])
    top_score = float(scores[raw_filter_idx])

    dist_top_prior = float(_circular_dist(np.array([top_eps]), eps_pred)[0])
    dist_chosen_prior = float(_circular_dist(np.array([chosen_eps]), eps_pred)[0])

    peak_idx = find_local_maxima(scores)

    mean_grid, valid_rows, order_centers, order_edges = build_echelle(f_masked, norm_power, numax, dnu, w)
    order_epsilons = order_by_order_consistency(mean_grid, valid_rows, delta)
    consistency_sigma = circular_std(order_epsilons)

    norm_log_power_echelle = bkg_normalized_log_power(f_masked, p_masked, bkg_params, nyq)
    mean_grid_echelle, _, _, _ = build_echelle(f_masked, norm_log_power_echelle, numax, dnu, w)

    result = dict(
        kic=kic, numax=numax, dnu=dnu, hwhm=hwhm, delta=delta,
        epsilon_p=float(chosen_eps),
        epsilon_p_scaled=float(epsilon_p_scaled),
        epsilon_p_edge_distance=float(epsilon_p_edge_distance),
        top_peak_epsilon=float(top_eps),
        top_peak_score=float(top_score),
        best_filter_score=float(best_filter_score),
        best_prior_weight=float(best_prior_weight),
        best_combined_score=float(best_combined_score),
        eps_pred_prior=float(eps_pred),
        prior_sigma=float(prior_sigma),
        prior_fwhm=float(prior_fwhm),
        dist_top_prior=float(dist_top_prior),
        dist_chosen_prior=float(dist_chosen_prior),
        n_peaks=int(len(peak_idx)),
        consistency_sigma=float(consistency_sigma) if not np.isnan(consistency_sigma) else np.nan,
        n_orders=len(order_centers),
        n_orders_valid=int(valid_rows.sum()),
        n_bins=N_BINS,
        mean_counts_per_bin=float(counts.mean()),
        min_counts_per_bin=int(counts.min()),
    )

    diag = dict(
        freq=freq, power=power, mask=mask,
        freq_masked=f_masked, norm_power=norm_power,
        collapsed_raw=collapsed_raw, collapsed=collapsed, counts=counts,
        scores=scores, scores_raw=scores_raw, peak_idx=peak_idx,
        prior_curve=prior_curve, combined_score=combined_score, filter_matrix=filt,
        mean_grid=mean_grid, mean_grid_echelle=mean_grid_echelle, valid_rows=valid_rows,
        order_centers=order_centers, order_edges=order_edges, order_epsilons=order_epsilons,
    )
    return result, diag


# ==========================================
# Diagnostic PDF
# ==========================================
def make_diagnostic_pdf(result: dict, diag: dict, plot_dir: str) -> str:
    os.makedirs(plot_dir, exist_ok=True)
    kic = result['kic']

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    fig.suptitle(
        f"KIC {kic} — epsilon_p diagnostics (chosen eps = {result['epsilon_p']:.4f})\n"
        f"numax = {result['numax']:.3f} uHz    dnu = {result['dnu']:.4f} uHz    "
        f"HWHM = {result['hwhm']:.3f} uHz",
        fontsize=12
    )

    order_edges = diag['order_edges']
    order_centers = diag['order_centers']
    order_epsilons = diag['order_epsilons']
    valid_rows = diag['valid_rows']

    ax = axes[0, 0]
    freq, power, mask = diag['freq'], diag['power'], diag['mask']
    ax.plot(freq[mask], power[mask], lw=0.6, color='0.3')
    ax.set_title(f"1. Windowed PSD, raw power (numax={result['numax']:.2f}, HWHM={result['hwhm']:.2f})")
    ax.set_xlabel("Frequency (uHz)")
    ax.set_ylabel("Power")

    ax = axes[0, 1]
    freq_masked, norm_power = diag['freq_masked'], diag['norm_power']
    ax.plot(freq_masked, norm_power, lw=0.4, color='darkslateblue')
    for e in order_edges:
        ax.axvline(e, color='gray', lw=0.4, ls=':', alpha=0.6)
    for lo, hi, oe, valid in zip(order_edges[:-1], order_edges[1:], order_epsilons, valid_rows):
        if valid:
            ax.text((lo + hi) / 2, 1.03, f"{oe:.2f}", ha='center', va='bottom',
                    fontsize=6, rotation=90, color='darkgreen')
    ax.set_ylim(0, 1.18)
    ax.set_title("2. Normalized PSD/bkg vs frequency")
    ax.set_xlabel("Frequency (uHz)  |  dotted lines = radial-order boundaries (width=dnu)")
    ax.set_ylabel("minmax PSD/bkg")

    ax = axes[1, 0]
    pcm = ax.pcolormesh(BIN_EDGES, order_edges, diag['mean_grid_echelle'],
                        shading='auto', cmap='viridis', vmin=0, vmax=1)
    if np.any(~valid_rows):
        for oc in order_centers[~valid_rows]:
            ax.axhline(oc, color='red', lw=0.4, alpha=0.3)
    ax.axvline(result['epsilon_p'], color='white', ls='--', lw=1.2)
    ax.set_title("3. Echelle diagram (minmax log10(PSD/bkg), display only)")
    ax.set_xlabel("Phase (f/dnu mod 1)")
    ax.set_ylabel("Frequency (uHz)")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="minmax log10(PSD/bkg)")

    ax = axes[1, 1]
    ax.plot(XCENTERS, diag['collapsed_raw'], color='navy', lw=1.2, label='collapsed minmax PSD/bkg')
    ax.axvline(result['epsilon_p'], color='crimson', ls='--', lw=1.5, label=f"chosen eps={result['epsilon_p']:.3f}")
    if abs(result['top_peak_epsilon'] - result['epsilon_p']) > 1e-9:
        ax.axvline(result['top_peak_epsilon'], color='orange', ls=':', lw=1.5,
                   label=f"filter-only peak eps={result['top_peak_epsilon']:.3f}")
    ax.axvline(result['eps_pred_prior'], color='gray', ls='-.', lw=1.2, label=f"prior eps_pred={result['eps_pred_prior']:.3f}")
    for lo, hi in circular_span(result['eps_pred_prior'], 0.30):
        ax.axvspan(lo, hi, color='gray', alpha=0.12)
    ax.set_title(f"4. Collapsed echelle (N_BINS={N_BINS})")
    ax.set_xlabel("Phase (x = (f/dnu) mod 1)")
    ax.set_ylabel("minmax PSD/bkg")
    ax.legend(fontsize=7, loc='upper right')

    ax = axes[2, 0]
    ax.plot(EPS_GRID, diag['scores'], lw=1.0, label='filter response')
    ax.plot(EPS_GRID, diag['prior_curve'], lw=1.0, ls='--', label='epsilon-dnu prior')
    ax.plot(EPS_GRID, diag['combined_score'], lw=1.4, label='filter × prior')
    ax.axvline(result['top_peak_epsilon'], color='orange', ls=':', lw=1.2,
               label=f'filter-only max={result["top_peak_epsilon"]:.3f}')
    ax.axvline(result['eps_pred_prior'], color='gray', ls='-.', lw=1.2, label=f'prior centre={result["eps_pred_prior"]:.3f}')
    ax.axvline(result['epsilon_p'], color='crimson', ls='--', lw=1.5, label=f'FINAL epsilon={result["epsilon_p"]:.3f}')
    ax.set_title("5. Filter response × epsilon-dnu prior")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Normalized response")
    ax.legend(fontsize=7, loc='best')

    ax = axes[2, 1]
    chosen_filter, filter_l0, filter_l2 = get_filter_at_epsilon(result['epsilon_p'], result['delta'])
    ax.plot(XCENTERS, filter_l0, lw=1.1, label='l=0 filter')
    ax.plot(XCENTERS, filter_l2, lw=1.1, label='l=2 filter')
    ax.plot(XCENTERS, chosen_filter, lw=1.8, label='combined filter')
    ax.set_title(f"6. Matched-filter structure at epsilon={result['epsilon_p']:.4f}")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Filter weight")
    ax.legend(fontsize=7, loc='best')

    ax = axes[3, 0]
    ax.axis('off')
    sigma_txt = f"{result['consistency_sigma']:.4f}" if not np.isnan(result['consistency_sigma']) else "n/a"
    lines = [
        "7. Diagnostic summary",
        f"numax = {result['numax']:.3f}   dnu = {result['dnu']:.4f}   HWHM = {result['hwhm']:.3f}",
        f"delta = {result['delta']:.4f}   n_orders = {result['n_orders']} (valid: {result['n_orders_valid']})",
        "",
        f"filter-only eps      = {result['top_peak_epsilon']:.4f}",
        f"prior eps_pred       = {result['eps_pred_prior']:.4f}",
        f"prior width param.   = {result['prior_fwhm']:.4f}",
        f"chosen eps_p         = {result['epsilon_p']:.4f}",
        f"distance to 0/1 edge = {result['epsilon_p_edge_distance']:.4f}",
        f"dist(chosen,prior)   = {result['dist_chosen_prior']:.4f}",
        "",
        f"filter response      = {result['best_filter_score']:.4f}",
        f"prior weight         = {result['best_prior_weight']:.4f}",
        f"product              = {result['best_combined_score']:.4f}",
        "",
        f"n_filter_peaks       = {result['n_peaks']}",
        f"consistency_sigma    = {sigma_txt}",
        f"mean counts/bin      = {result['mean_counts_per_bin']:.1f}",
        f"min counts/bin       = {result['min_counts_per_bin']}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            va='top', ha='left', family='monospace')

    ax = axes[3, 1]
    counts = diag['counts']
    ax.bar(XCENTERS, counts, width=(1.0 / N_BINS) * 0.9, color='slategray')
    ax.axhline(counts.mean(), color='crimson', ls='--', lw=1, label=f"mean = {counts.mean():.1f} pts/bin")
    ax.set_title(f"8. Points per phase bin (N_BINS={N_BINS})")
    ax.set_xlabel("Phase (x = (f/dnu) mod 1)")
    ax.set_ylabel("Frequency samples per bin")
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(plot_dir, f"{kic}_epp_diagnostics.pdf")
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="SPANDA-Stage 2a: estimate epsilon_p for one star.")
    parser.add_argument("kic", type=int, help="KIC number")
    parser.add_argument("--stage1-csv", required=True, help="Stage 1 summary CSV (stage1_summary_<N>.csv)")
    parser.add_argument("--psd-dir", required=True, help="folder containing the power spectra")
    parser.add_argument("--psd-pattern", default="{kic}.npy", help="PSD file name with {kic} placeholder (default: {kic}.npy)")
    parser.add_argument("--outdir", default="./epsilon_p_estimates", help="folder for <KIC>_epp_estimates.npy")
    parser.add_argument("--plot", action="store_true", help="save a diagnostic PDF")
    parser.add_argument("--plot-dir", default="./stage2a_plots", help="folder for the diagnostic PDF")
    parser.add_argument("--no-file", dest="no_file", action="store_true", help="do not save the .npy file")
    args = parser.parse_args()

    result, diag = estimate_epsilon_p(args.kic, args.stage1_csv, args.psd_dir, args.psd_pattern)

    print(f"KIC {result['kic']}: numax = {result['numax']:.3f} uHz, "
          f"dnu = {result['dnu']:.4f} uHz, HWHM = {result['hwhm']:.3f} uHz")

    if args.no_file:
        print("`--no-file` given: estimates .npy not saved.")
    else:
        print(f"Estimates saved to: {save_estimates(result, diag, args.outdir)}")

    if args.plot:
        print(f"Diagnostic PDF saved to: {make_diagnostic_pdf(result, diag, args.plot_dir)}")

    sigma_txt = f"{result['consistency_sigma']:.4f}" if not np.isnan(result['consistency_sigma']) else "n/a"
    print(f"KIC {result['kic']}: epsilon_p = {result['epsilon_p']:.4f} "
          f"(distance from nearest 0/1 edge = {result['epsilon_p_edge_distance']:.4f}, "
          f"n_peaks={result['n_peaks']}, consistency_sigma={sigma_txt})")


if __name__ == "__main__":
    main()
