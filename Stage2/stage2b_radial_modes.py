#!/usr/bin/env python3
"""
SPANDA-Stage 2b: local identification of the radial (l=0) modes of one star and
initial estimates of their frequencies and heights.

Usage:
    python3 stage2b_radial_modes.py <KIC> --stage1-csv <CSV> --psd-dir <DIR>
                                    [--epsilon-summary FILE] [--plot] [--alias] [--no-file]

Authors: Sapnodip Pramanik, Shatanik Bhattacharya
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def compute_matched_filter_matrix(phase_centers, delta_phase, sigma_phase, inverted=False, signed_phase=False):
    """Circulant two-peak (l=0, l=2) matched-filter matrix."""
    shift = delta_phase if inverted else -delta_phase
    x_grid, eps_grid = np.meshgrid(phase_centers, phase_centers)

    dist_p = np.abs(x_grid - eps_grid)
    dist_p = np.minimum(dist_p, 1.0 - dist_p)

    if signed_phase:
        eps_comp = (eps_grid + shift + 0.5) % 1.0 - 0.5
    else:
        eps_comp = (eps_grid + shift) % 1.0

    dist_c = np.abs(x_grid - eps_comp)
    dist_c = np.minimum(dist_c, 1.0 - dist_c)

    filt = np.exp(-0.5 * (dist_p / sigma_phase) ** 2) + 0.7 * np.exp(-0.5 * (dist_c / sigma_phase) ** 2)
    filt /= np.sum(filt, axis=1, keepdims=True)
    return filt


def load_psd(psd_dir, pattern, kic):
    path = os.path.join(psd_dir, pattern.format(kic=kic))
    if not os.path.exists(path):
        raise FileNotFoundError(f"PSD file not found: {path}")
    arr = np.load(path)
    if arr.shape[0] < arr.shape[1]:
        return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float)
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)


def main(kic, stage1_csv, epsilon_summary, psd_dir, psd_pattern, outdir, plot_dir,
         gen_plot=False, gen_alias=False, gen_file=True):
    # =========================================================================
    # 1. Load data
    # =========================================================================
    data = np.loadtxt(epsilon_summary, skiprows=1, ndmin=2)
    kic_indices = np.where(data[:, 0] == kic)[0]
    if len(kic_indices) == 0:
        raise ValueError(f"KIC {kic} not found in the Stage 2a summary table: {epsilon_summary}")
    i = kic_indices[0]
    print(f"Loaded Stage 2a values for KIC {kic}:\n", data[i])

    freq_full, power_full = load_psd(psd_dir, psd_pattern, kic)
    sort_idx = np.argsort(freq_full)
    freq_full = freq_full[sort_idx]
    power_full = power_full[sort_idx]

    nu_nyquist = freq_full[-1]
    nu_max    = data[i][2]
    dnu       = data[i][3]
    HWHM      = data[i][4]
    epsilon_p = data[i][5]
    beta      = 0.076

    # Signed phase representation (-0.5, 0.5) when epsilon is close to 0 or 1
    EPS_WRAP_TOL = 0.10
    use_signed_phase = (epsilon_p <= EPS_WRAP_TOL) or (epsilon_p >= 1.0 - EPS_WRAP_TOL)
    epsilon_phase_ref = ((epsilon_p + 0.5) % 1.0) - 0.5 if use_signed_phase else epsilon_p

    def fold_phase(freq):
        phase = (np.asarray(freq, dtype=np.float64) / dnu) % 1.0
        if use_signed_phase:
            phase = (phase + 0.5) % 1.0 - 0.5
        return phase

    df_bkg = pd.read_csv(stage1_csv, low_memory=False)
    df_bkg.columns = df_bkg.columns.str.strip()
    bkg_row = df_bkg[df_bkg['kic'] == kic]
    if bkg_row.empty:
        raise ValueError(f"KIC {kic} not found in Stage 1 summary CSV: {stage1_csv}")

    bkg_cols = [
        "log_10_a_1_over_b_1_map", "log_10_b_1_map", "log_10_a_2_map",
        "log_10_a_3_map", "log_10_b_2_map", "log_10_b_3_map", "log_10_W_map"
    ]
    bkg_params = bkg_row[bkg_cols].values[0].astype(np.float64)

    # Smoothed PSD for peak location and heights
    Gamma = 0.026 + 0.12 * np.log10(dnu)
    freq_range = freq_full[-1] - freq_full[0]
    filter_width = max(int(len(power_full) * (Gamma / 2.0) / freq_range), 1) if (freq_range > 0 and Gamma > 0) else 1

    if filter_width > 1:
        kernel = np.ones(filter_width, dtype=np.float64) / filter_width
        smooth_psd = np.convolve(power_full, kernel, mode='same')
    else:
        smooth_psd = power_full.copy()

    # Reference radial mode
    epsilon_p_work = epsilon_phase_ref
    n_max = nu_max / dnu - epsilon_p_work
    n_ref = int(np.round(n_max))
    nu_p_nl = (n_ref + epsilon_p_work + beta * (n_ref - n_max) ** 2 / (2.0 * n_max)) * dnu

    freq_min_local = nu_p_nl - dnu / 3.0
    freq_max_local = nu_p_nl + dnu / 3.0
    half_width_global = 3.0 * max(HWHM, dnu)
    freq_min_global = nu_max - half_width_global
    freq_max_global = nu_max + half_width_global

    d02 = 0.121 * dnu + 0.047
    delta_phase = d02 / dnu

    idx_g0, idx_g1 = np.searchsorted(freq_full, [freq_min_global, freq_max_global])
    freq_global = freq_full[idx_g0:idx_g1]
    power_global = power_full[idx_g0:idx_g1]

    valid_g = power_global > 0
    freq_global = freq_global[valid_g]
    power_global = power_global[valid_g]

    bkg_global = eval_bkg_model(freq_global, bkg_params, nu_nyquist)
    power_snr_global = power_global / bkg_global

    print("=" * 65)
    print(f"KIC {kic} | Nyquist Frequency: {nu_nyquist:.4f} µHz")
    print(f"Global Envelope: [{freq_min_global:.4f}, {freq_max_global:.4f}] µHz")
    print(f"Flags: gen_plot={gen_plot}, gen_alias={gen_alias}, gen_file={gen_file}")
    print("=" * 65)

    # =========================================================================
    # 2. Matched-filter matrices
    # =========================================================================
    num_bins = 100
    if use_signed_phase:
        phase_edges = np.linspace(-0.5, 0.5, num_bins + 1)
    else:
        phase_edges = np.linspace(0.0, 1.0, num_bins + 1)
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    bin_width = phase_edges[1] - phase_edges[0]

    sigma_phase = (delta_phase / 6.0) / np.sqrt(2.0 * np.log(2.0))
    if sigma_phase <= 0 or np.isnan(sigma_phase):
        sigma_phase = 0.05

    filt_matrix = compute_matched_filter_matrix(phase_centers, delta_phase, sigma_phase,
                                                inverted=False, signed_phase=use_signed_phase)
    filt_matrix_inv = (compute_matched_filter_matrix(phase_centers, delta_phase, sigma_phase,
                                                     inverted=True, signed_phase=use_signed_phase)
                       if gen_alias else None)

    # Global collapsed profile (diagnostic plot only)
    phase_global_norm = fold_phase(freq_global)
    collapsed_power_global, _ = np.histogram(phase_global_norm, bins=phase_edges, weights=power_snr_global)
    std_g = np.std(collapsed_power_global)
    collapsed_norm_g = (collapsed_power_global - np.mean(collapsed_power_global)) / (std_g if std_g > 0 else 1.0)
    scores_global = filt_matrix @ collapsed_norm_g
    best_eps_global = phase_centers[np.argmax(scores_global)]
    comp_eps_global = (
        (best_eps_global - delta_phase + 0.5) % 1.0 - 0.5
        if use_signed_phase else
        (best_eps_global - delta_phase) % 1.0
    )

    # Reference local window
    idx_l0, idx_l1 = np.searchsorted(freq_full, [freq_min_local, freq_max_local])
    freq_local = freq_full[idx_l0:idx_l1]
    power_local = power_full[idx_l0:idx_l1]
    valid_l = power_local > 0
    freq_local, power_local = freq_local[valid_l], power_local[valid_l]

    bkg_local = eval_bkg_model(freq_local, bkg_params, nu_nyquist)
    power_snr_local = power_local / bkg_local

    phase_local_norm = fold_phase(freq_local)
    collapsed_power_local, _ = np.histogram(phase_local_norm, bins=phase_edges, weights=power_snr_local)

    std_l = np.std(collapsed_power_local)
    collapsed_norm_l = (collapsed_power_local - np.mean(collapsed_power_local)) / (std_l if std_l > 0 else 1.0)
    scores_local = filt_matrix @ collapsed_norm_l
    best_idx_l = np.argmax(scores_local)
    best_eps_local = phase_centers[best_idx_l]
    peak_score_local = scores_local[best_idx_l]

    k_ref = int(np.round((nu_p_nl / dnu) - best_eps_local))
    nu_l0_measured_ref = (k_ref + best_eps_local) * dnu
    nu_l2_measured_ref = nu_l0_measured_ref - d02
    n_ref = k_ref

    # Reference mode: peak of the smoothed PSD near the filter frequency, and height
    sigma_freq = sigma_phase * dnu
    nu_l0_ref_filter = nu_l0_measured_ref
    ref_search_min = max(nu_l0_ref_filter - 2.0 * sigma_freq, freq_min_local)
    ref_search_max = min(nu_l0_ref_filter + 2.0 * sigma_freq, freq_max_local)
    smooth_local_ref = smooth_psd[idx_l0:idx_l1][valid_l]
    ref_search_mask = (freq_local >= ref_search_min) & (freq_local <= ref_search_max)
    if np.any(ref_search_mask):
        ref_peak_idx = np.argmax(smooth_local_ref[ref_search_mask])
        ref_peak_freqs = freq_local[ref_search_mask]
        nu_l0_measured_ref = float(ref_peak_freqs[ref_peak_idx])
        pwr_at_ref = float(smooth_local_ref[ref_search_mask][ref_peak_idx])
    else:
        pwr_at_ref = float(np.interp(nu_l0_measured_ref, freq_full, smooth_psd))
    nu_l2_measured_ref = nu_l0_measured_ref - d02
    bkg_at_ref = float(eval_bkg_model(nu_l0_measured_ref, bkg_params, nu_nyquist))
    height_l0_ref = max(pwr_at_ref - bkg_at_ref, 1e-12)

    # =========================================================================
    # 3. Asymptotic grid of radial orders
    # =========================================================================
    beta_p = 0.076
    epsilon_p_updated = best_eps_local
    n_max_updated = (nu_max / dnu) - epsilon_p_updated

    n_search = np.arange(int(np.floor(n_max_updated - 10)), int(np.ceil(n_max_updated + 10)))
    nu_l0_guesses_all = (
        n_search + epsilon_p_updated + (beta_p / (2.0 * n_max_updated)) * (n_search - n_max_updated) ** 2
    ) * dnu

    is_direct = (nu_l0_guesses_all <= nu_nyquist) & (nu_l0_guesses_all >= freq_min_global) & (nu_l0_guesses_all <= freq_max_global)
    if gen_alias:
        nu_alias_all = 2.0 * nu_nyquist - nu_l0_guesses_all
        is_aliased_valid = (nu_l0_guesses_all > nu_nyquist) & (nu_alias_all >= freq_min_global) & (nu_alias_all <= freq_max_global)
    else:
        is_aliased_valid = np.zeros_like(nu_l0_guesses_all, dtype=bool)

    valid_mask = is_direct | is_aliased_valid
    n_guesses = n_search[valid_mask]
    nu_l0_guesses = nu_l0_guesses_all[valid_mask]

    # =========================================================================
    # 4. Local windows ("chunks") and matched filtering for the other orders
    # =========================================================================
    all_measured_modes = {
        n_ref: {'l0': nu_l0_measured_ref, 'H_n0': height_l0_ref, 'l2': nu_l2_measured_ref,
                'score': peak_score_local, 'eps': best_eps_local, 'is_aliased': False}
    }

    other_mask = (n_guesses != n_ref)
    other_orders = n_guesses[other_mask]
    other_nu_guesses = nu_l0_guesses[other_mask]
    n_chunks = len(other_orders)

    print(f"\nFitting {n_chunks} other asymptotic chunks (gen_alias={gen_alias})...\n")
    print(f"{'Order (n)':<10} | {'Type':<10} | {'l=0 (µHz)':<12} | {'H_n0':<12} | {'l=2 (µHz)':<12} | {'Score':<8} | {'eps':<6}")
    print("-" * 88)

    chunk_data = []

    if n_chunks > 0:
        is_alias_arr = other_nu_guesses > nu_nyquist
        delta_k = np.where(is_alias_arr, other_nu_guesses - nu_nyquist, 0.0)
        f_targets = np.where(is_alias_arr, nu_nyquist - delta_k, other_nu_guesses)

        f_mins = np.where(is_alias_arr, f_targets - (d02 / 3.0), f_targets - 1.5 * d02)
        f_maxs = np.where(is_alias_arr, f_targets + 1.5 * d02, f_targets + (d02 / 2.0))

        idx_starts = np.searchsorted(freq_full, f_mins)
        idx_ends   = np.searchsorted(freq_full, f_maxs)

        for j in range(n_chunks):
            order_i = other_orders[j]
            is_al = is_alias_arr[j]
            f_target = f_targets[j]
            mode_type = "Aliased" if is_al else "Standard"

            s_idx, e_idx = idx_starts[j], idx_ends[j]
            f_c = freq_full[s_idx:e_idx]
            p_c = power_full[s_idx:e_idx]
            valid_pts = p_c > 0
            f_c, p_c = f_c[valid_pts], p_c[valid_pts]

            if len(f_c) < 3:
                continue

            bkg_c = eval_bkg_model(f_c, bkg_params, nu_nyquist)
            p_snr_c = p_c / bkg_c

            phases = fold_phase(f_c)
            pwr_hist, _ = np.histogram(phases, bins=phase_edges, weights=p_snr_c)

            std_c = np.std(pwr_hist)
            norm_hist = (pwr_hist - np.mean(pwr_hist)) / (std_c if std_c > 0 else 1.0)

            active_filt = filt_matrix_inv if is_al else filt_matrix
            scores = active_filt @ norm_hist
            b_idx = np.argmax(scores)
            best_eps_c = phase_centers[b_idx]
            score_c = scores[b_idx]

            nu_l0_c = (np.round((f_target / dnu) - best_eps_c) + best_eps_c) * dnu
            nu_l0_filter_c = nu_l0_c
            search_min_c = max(nu_l0_filter_c - 2.0 * sigma_freq, f_c[0])
            search_max_c = min(nu_l0_filter_c + 2.0 * sigma_freq, f_c[-1])
            smooth_c = smooth_psd[s_idx:e_idx][valid_pts]
            search_mask_c = (f_c >= search_min_c) & (f_c <= search_max_c)
            if np.any(search_mask_c):
                peak_idx_c = np.argmax(smooth_c[search_mask_c])
                peak_freqs_c = f_c[search_mask_c]
                nu_l0_c = float(peak_freqs_c[peak_idx_c])
                pwr_at_l0 = float(smooth_c[search_mask_c][peak_idx_c])
            else:
                pwr_at_l0 = float(np.interp(nu_l0_c, freq_full, smooth_psd))
            nu_l2_c = nu_l0_c + d02 if is_al else nu_l0_c - d02
            bkg_at_l0 = float(eval_bkg_model(nu_l0_c, bkg_params, nu_nyquist))
            height_l0_c = max(pwr_at_l0 - bkg_at_l0, 1e-12)

            all_measured_modes[order_i] = {'l0': nu_l0_c, 'H_n0': height_l0_c, 'l2': nu_l2_c,
                                           'score': score_c, 'eps': best_eps_c, 'is_aliased': is_al}

            print(f"{order_i:<10} | {mode_type:<10} | {nu_l0_c:<12.4f} | {height_l0_c:<12.4e} | {nu_l2_c:<12.4f} | {score_c:<8.2f} | {best_eps_c:<6.3f}")

            if gen_plot:
                chunk_data.append((order_i, is_al, f_target, f_mins[j], f_maxs[j], f_c, p_c, smooth_c,
                                   nu_l0_filter_c, search_min_c, search_max_c, nu_l0_c, nu_l2_c, score_c, best_eps_c))

    # =========================================================================
    # 5. Diagnostic plots (optional)
    # =========================================================================
    if gen_plot:
        orders = np.floor(freq_global / dnu).astype(int)
        nmin, nmax = orders.min(), orders.max()
        n_orders = nmax - nmin + 1
        nbins_x = 300
        xgrid = np.linspace(-0.5, 0.5, nbins_x + 1) if use_signed_phase else np.linspace(0.0, 1.0, nbins_x + 1)
        xcenters = 0.5 * (xgrid[:-1] + xgrid[1:])
        E = np.full((n_orders, nbins_x), np.nan)

        for ord_val in range(nmin, nmax + 1):
            mask_o = (orders == ord_val)
            if np.count_nonzero(mask_o) < 2:
                continue
            x_o = phase_global_norm[mask_o]
            p_o = np.log10(power_snr_global[mask_o])
            s = np.argsort(x_o)
            x_padded = np.concatenate([x_o[s] - 1.0, x_o[s], x_o[s] + 1.0])
            p_padded = np.concatenate([p_o[s], p_o[s], p_o[s]])
            E[ord_val - nmin] = np.interp(xcenters, x_padded, p_padded)

        emin, emax = np.nanmin(E), np.nanmax(E)
        if emax > emin:
            E = (E - emin) / (emax - emin)
        yedges = np.arange(nmin - 0.5, nmax + 1.5) * dnu

        os.makedirs(plot_dir, exist_ok=True)
        pdf_path = os.path.join(plot_dir, f'{kic}_diagnostic_plots.pdf')
        pdf = PdfPages(pdf_path)

        phase_grid = np.linspace(-0.5, 0.5, 1000) if use_signed_phase else np.linspace(0.0, 1.0, 1000)
        dist_p_g = np.minimum(np.abs(phase_grid - best_eps_global), 1.0 - np.abs(phase_grid - best_eps_global))
        dist_c_g = np.minimum(np.abs(phase_grid - comp_eps_global), 1.0 - np.abs(phase_grid - comp_eps_global))
        template_phase_scaled = (
            np.exp(-0.5 * (dist_p_g / sigma_phase) ** 2) + 0.7 * np.exp(-0.5 * (dist_c_g / sigma_phase) ** 2)
        ) * np.max(collapsed_power_global)

        fig1 = plt.figure(figsize=(10, 15), dpi=120)
        gs = fig1.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 1.1, 1.7], hspace=0.3)
        phase_xlabel = (r'Signed Folded Phase $((\nu/\Delta\nu)\ \mathrm{mod}\ 1)$'
                        if use_signed_phase else r'Folded Phase $(\nu \ \mathrm{mod} \ \Delta\nu) / \Delta\nu$')

        ax1 = fig1.add_subplot(gs[0])
        ax1.bar(phase_centers, collapsed_power_global, width=bin_width, color='gray', alpha=0.6, edgecolor='black', lw=0.4, label='Folded SNR (PSD/Bkg)')
        ax1.plot(phase_grid, template_phase_scaled, color='crimson', lw=1.8, linestyle='--', label='Global Filter Template')
        ax1.axvline(best_eps_global, color='darkorange', linestyle=':', lw=1.5, label=rf'Global $l=0$ ($\epsilon = {best_eps_global:.3f}$)')
        ax1.axvline(comp_eps_global, color='royalblue', linestyle=':', lw=1.5, label=r'Global $l=2$')
        ax1.set_xlim(-0.5, 0.5) if use_signed_phase else ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel(phase_xlabel, fontsize=10)
        ax1.set_ylabel('Folded SNR', fontsize=10)
        ax1.set_title(f'KIC {kic} — Global Collapsed Échelle & Filter', fontsize=11)
        ax1.legend(loc='upper right', fontsize=8, frameon=True)
        ax1.grid(True, linestyle='--', alpha=0.4)

        ax2 = fig1.add_subplot(gs[1])
        freq_grid_local = np.linspace(freq_min_local, freq_max_local, 2000)
        template_chunk = (
            np.exp(-0.5 * ((freq_grid_local - nu_l0_measured_ref) / sigma_freq) ** 2)
            + 0.7 * np.exp(-0.5 * ((freq_grid_local - nu_l2_measured_ref) / sigma_freq) ** 2)
        ) * np.max(power_local)
        ax2.plot(freq_local, power_local, color='black', lw=1.0, alpha=0.75, label='Observed PSD Chunk')
        ax2.fill_between(freq_local, power_local, color='gray', alpha=0.25)
        ax2.plot(freq_local, smooth_local_ref, color='forestgreen', lw=1.5, label='Smoothed PSD')
        ax2.axvspan(ref_search_min, ref_search_max, color='gold', alpha=0.25, label=r'$l=0$ peak search ($\pm2\sigma$)')
        ax2.plot(freq_grid_local, template_chunk, color='crimson', lw=1.8, linestyle='--', label=f'Local Template (Score = {peak_score_local:.2f})')
        ax2.axvline(nu_l0_measured_ref, color='darkorange', linestyle=':', lw=1.8, label=rf'Final $l=0$ ({nu_l0_measured_ref:.3f} $\mu\mathrm{{Hz}}$)')
        ax2.axvline(nu_l2_measured_ref, color='royalblue', linestyle='--', lw=1.8, label=rf'Measured $l=2$ ({nu_l2_measured_ref:.3f} $\mu\mathrm{{Hz}}$)')
        ax2.set_xlim(freq_min_local, freq_max_local)
        ax2.set_ylim(0, np.max(power_local) * 1.18)
        ax2.set_xlabel(r'Frequency ($\mu\mathrm{Hz}$)', fontsize=10)
        ax2.set_ylabel(r'PSD ($\mathrm{ppm}^2/\mu\mathrm{Hz}$)', fontsize=10)
        ax2.set_title(f'Reference Chunk PSD at $n={n_ref}$', fontsize=11)
        ax2.legend(loc='upper right', fontsize=8, frameon=True)
        ax2.grid(True, linestyle='--', alpha=0.4)

        ax3 = fig1.add_subplot(gs[2])
        ax3.plot(freq_global, power_global, color='black', lw=0.9, alpha=0.7, label='Observed PSD (Global)')
        ax3.fill_between(freq_global, power_global, color='gray', alpha=0.15)
        ax3.axvspan(freq_min_local, freq_max_local, color='gold', alpha=0.22, label='Reference Window')
        if nu_nyquist <= freq_max_global:
            ax3.axvline(nu_nyquist, color='darkviolet', linestyle='-', lw=2.0, label=rf'Nyquist ({nu_nyquist:.2f} $\mu\mathrm{{Hz}}$)')
        for idx, (order, nu_guess) in enumerate(zip(n_guesses, nu_l0_guesses)):
            is_alias = nu_guess > nu_nyquist
            nu_plot = (2.0 * nu_nyquist - nu_guess) if is_alias else nu_guess
            tag = f'n={order} (Alias)' if is_alias else f'n={order}'
            col = 'purple' if is_alias else 'crimson'
            ax3.axvline(nu_plot, color=col, linestyle='--', lw=1.5, label=r'Asymptotic $l=0$' if idx == 0 else None)
            ax3.text(nu_plot, np.max(power_global) * 0.93, tag, rotation=90, va='top', ha='right', fontsize=7.5, color=col, weight='bold')
        ax3.set_xlim(freq_min_global, freq_max_global)
        ax3.set_ylim(0, np.max(power_global) * 1.15)
        ax3.set_xlabel(r'Frequency ($\mu\mathrm{Hz}$)', fontsize=10)
        ax3.set_ylabel(r'PSD ($\mathrm{ppm}^2/\mu\mathrm{Hz}$)', fontsize=10)
        ax3.set_title(r'Global PSD with Predicted Asymptotic $l=0$ Modes', fontsize=11)
        ax3.legend(loc='upper right', fontsize=8, frameon=True)
        ax3.grid(True, linestyle='--', alpha=0.4)

        ax4 = fig1.add_subplot(gs[3])
        mesh = ax4.pcolormesh(xgrid, yedges, E, cmap='rainbow', shading='auto')
        ax4.axvline(best_eps_global, color='k', linestyle='--', lw=1.4, label=rf'Global Track ($\epsilon={best_eps_global:.3f}$)')
        ax4.axvline(best_eps_local, color='yellow', linestyle='-', lw=1.6, label=rf'Local Track ($\epsilon={best_eps_local:.3f}$)')
        ax4.axhspan(freq_min_local, freq_max_local, color='black', alpha=0.25, label='Reference Window')
        cbar = fig1.colorbar(mesh, ax=ax4, orientation='horizontal', pad=0.18, aspect=35)
        cbar.set_label(r'Normalized $\log_{10}(\mathrm{PSD / Bkg})$', fontsize=9.5)
        ax4.set_xlim(-0.5, 0.5) if use_signed_phase else ax4.set_xlim(0.0, 1.0)
        ax4.set_ylim(yedges[0], yedges[-1])
        ax4.set_xlabel(phase_xlabel, fontsize=10)
        ax4.set_ylabel(r'Frequency ($\mu\mathrm{Hz}$)', fontsize=10)
        ax4.set_title(r'2D Échelle Diagram (Background Normalized)', fontsize=11)
        ax4.legend(loc='upper right', fontsize=8, frameon=True)
        ax4.grid(True, linestyle='--', color='white', alpha=0.25)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        fig_snr, ax_snr = plt.subplots(figsize=(12, 5.5), dpi=120)
        ax_snr.plot(freq_global, power_snr_global, color='black', lw=0.9, alpha=0.75, label=r'Observed $\mathrm{PSD / Bkg}$')
        ax_snr.axhline(1.0, color='crimson', linestyle='--', lw=1.3, label='Background Baseline (SNR = 1.0)')
        if nu_nyquist <= freq_max_global:
            ax_snr.axvline(nu_nyquist, color='darkviolet', linestyle='-', lw=2.0, label=rf'Nyquist ({nu_nyquist:.2f} $\mu\mathrm{{Hz}}$)')
        ax_snr.set_xlim(freq_min_global, freq_max_global)
        ax_snr.set_ylim(0, np.percentile(power_snr_global, 99.8) * 1.25)
        ax_snr.set_xlabel(r'Frequency ($\mu\mathrm{Hz}$)', fontsize=11)
        ax_snr.set_ylabel(r'$\mathrm{PSD / Bkg}$', fontsize=11)
        ax_snr.set_title(f'KIC {kic} — Background-Normalized Power Spectrum', fontsize=12)
        ax_snr.legend(loc='upper right', fontsize=8.5, frameon=True)
        ax_snr.grid(True, linestyle='--', alpha=0.4)
        fig_snr.tight_layout()
        pdf.savefig(fig_snr)
        plt.close(fig_snr)

        fig_final, ax_fin = plt.subplots(figsize=(12, 6), dpi=120)
        ax_fin.plot(freq_global, power_global, color='black', lw=0.9, alpha=0.75, label='Observed PSD (Global)')
        ax_fin.fill_between(freq_global, power_global, color='gray', alpha=0.15)
        ax_fin.plot(freq_global, bkg_global, color='crimson', lw=1.5, linestyle='-', label='Granulation + Noise Background')
        if nu_nyquist <= freq_max_global:
            ax_fin.axvline(nu_nyquist, color='darkviolet', linestyle='-', lw=2.0, label=rf'Nyquist Cutoff ({nu_nyquist:.2f} $\mu\mathrm{{Hz}}$)')
        max_pwr_g = np.max(power_global)
        for k, order_k in enumerate(sorted(all_measured_modes.keys())):
            l0_v = all_measured_modes[order_k]['l0']
            l2_v = all_measured_modes[order_k]['l2']
            is_al = all_measured_modes[order_k]['is_aliased']
            col_l0 = 'darkviolet' if is_al else 'darkorange'
            col_l2 = 'magenta' if is_al else 'royalblue'
            lbl_l0 = (r'Refined $l=0$ modes' if not is_al else r'Aliased $l=0$ modes') if (k == 0 or is_al) else None
            lbl_l2 = (r'Refined $l=2$ modes' if not is_al else r'Aliased $l=2$ modes') if (k == 0 or is_al) else None
            ax_fin.axvline(l0_v, color=col_l0, linestyle=':', lw=1.8, label=lbl_l0)
            ax_fin.axvline(l2_v, color=col_l2, linestyle=':', lw=1.5, label=lbl_l2)
            tag_n = f'n={order_k}' + (' (Al)' if is_al else '')
            if freq_min_global <= l0_v <= freq_max_global:
                ax_fin.text(l0_v, max_pwr_g * 0.94, f'{tag_n}\n(l=0)', rotation=90, va='top', ha='center', fontsize=7.5, color=col_l0, weight='bold')
            if freq_min_global <= l2_v <= freq_max_global:
                ax_fin.text(l2_v, max_pwr_g * 0.80, 'l=2', rotation=90, va='top', ha='center', fontsize=7.0, color=col_l2, weight='bold')
        ax_fin.axvspan(freq_min_local, freq_max_local, color='gold', alpha=0.18, label=rf'Reference Chunk $n={n_ref}$ ($\pm\Delta\nu/3$)')
        ax_fin.set_xlim(freq_min_global, freq_max_global)
        ax_fin.set_ylim(0, max_pwr_g * 1.18)
        ax_fin.set_xlabel(r'Frequency ($\mu\mathrm{Hz}$)', fontsize=11)
        ax_fin.set_ylabel(r'PSD ($\mathrm{ppm}^2/\mu\mathrm{Hz}$)', fontsize=11)
        ax_fin.set_title(f'KIC {kic} — Spectrum with Refined $l=0$ and $l=2$ Modes Marked', fontsize=12)
        ax_fin.legend(loc='upper right', fontsize=8.5, frameon=True)
        ax_fin.grid(True, linestyle='--', alpha=0.4)
        fig_final.tight_layout()
        pdf.savefig(fig_final)
        plt.close(fig_final)

        if chunk_data:
            n_cols = 2
            n_rows = int(np.ceil(len(chunk_data) / n_cols))
            fig_chunks, axes_chunks = plt.subplots(n_rows, n_cols, figsize=(13, 3.4 * n_rows), dpi=120)
            axes_chunks = np.atleast_1d(axes_chunks).flatten()
            for c_idx, entry in enumerate(chunk_data):
                (ord_i, is_al, f_targ, f_min_c, f_max_c, f_c, p_c, smooth_c_plot, l0_filter_c_plot,
                 search_min_plot, search_max_plot, l0_c, l2_c, sc_c, eps_c) = entry
                ax_c = axes_chunks[c_idx]
                fev = np.linspace(f_min_c, f_max_c, 1000)
                templ = (
                    np.exp(-0.5 * ((fev - l0_filter_c_plot) / sigma_freq) ** 2)
                    + 0.7 * np.exp(-0.5 * ((fev - (l0_filter_c_plot - d02)) / sigma_freq) ** 2)
                ) * np.max(p_c)
                ax_c.plot(f_c, p_c, color='black', lw=1.0, alpha=0.75, label='Observed PSD')
                ax_c.fill_between(f_c, p_c, color='gray', alpha=0.2)
                ax_c.plot(f_c, smooth_c_plot, color='forestgreen', lw=1.4, label='Smoothed PSD')
                ax_c.axvspan(search_min_plot, search_max_plot, color='gold', alpha=0.25, label=r'$l=0$ peak search ($\pm2\sigma$)')
                ax_c.plot(fev, templ, color='purple' if is_al else 'crimson', lw=1.6, linestyle='--', label=f'Template (Score={sc_c:.2f})')
                ax_c.axvline(f_targ, color='forestgreen', linestyle=':', lw=1.5, label=rf'Guess ({f_targ:.3f} $\mu\mathrm{{Hz}}$)')
                ax_c.axvline(l0_filter_c_plot, color='gray', linestyle='--', lw=1.3, label=rf'Filter $l=0$ ({l0_filter_c_plot:.3f} $\mu\mathrm{{Hz}}$)')
                ax_c.axvline(l0_c, color='darkorange', linestyle=':', lw=1.7, label=rf'Final $l=0$ ({l0_c:.3f} $\mu\mathrm{{Hz}}$)')
                ax_c.axvline(l2_c, color='royalblue', linestyle='--', lw=1.7, label=rf'Fit $l=2$ ({l2_c:.3f} $\mu\mathrm{{Hz}}$)')
                ax_c.set_xlim(f_min_c, f_max_c)
                ax_c.set_ylim(0, np.max(p_c) * 1.25)
                ax_c.set_xlabel(r'Frequency ($\mu\mathrm{Hz}$)', fontsize=9.5)
                ax_c.set_ylabel(r'PSD ($\mathrm{ppm}^2/\mu\mathrm{Hz}$)', fontsize=9.5)
                ax_c.set_title(rf'Radial Order $n={ord_i}$ ($\epsilon={eps_c:.3f}$){" [ALIASED]" if is_al else ""}', fontsize=10)
                ax_c.legend(loc='upper left', ncol=2, fontsize=7.2, frameon=True)
                ax_c.grid(True, linestyle='--', alpha=0.4)
            for j in range(c_idx + 1, len(axes_chunks)):
                fig_chunks.delaxes(axes_chunks[j])
            fig_chunks.tight_layout()
            pdf.savefig(fig_chunks)
            plt.close(fig_chunks)

        pdf.close()
        print(f"Saved diagnostic plots to PDF: {pdf_path}")

    # =========================================================================
    # 6. Output, sorted by l=0 frequency
    # =========================================================================
    sorted_by_freq = sorted(all_measured_modes.items(), key=lambda item: item[1]['l0'])
    freqs_l0 = np.array([m[1]['l0'] for m in sorted_by_freq], dtype=float)
    heights_l0 = np.array([m[1]['H_n0'] for m in sorted_by_freq], dtype=float)
    eps_p = np.array([m[1]['eps'] for m in sorted_by_freq], dtype=float)
    orders_arr = np.array([m[0] for m in sorted_by_freq], dtype=int)
    alias_flags = np.array([m[1]['is_aliased'] for m in sorted_by_freq], dtype=bool)

    if gen_file:
        os.makedirs(outdir, exist_ok=True)
        output_path = os.path.join(outdir, f'{kic}_p_frequencies.npy')
        np.save(output_path, {
            'kic': kic, 'radial_orders': orders_arr, 'nu_l0': freqs_l0,
            'H_n0': heights_l0, 'epsilon_p': eps_p, 'is_aliased': alias_flags
        })
        print(f"\nSaved {len(freqs_l0)} mode measurements to: {output_path}")

    print(f"\nFinal Extracted Mode Table for KIC {kic} (Sorted by l=0 Frequency):")
    print(f"{'Order (n)':<10} | {'l=0 (µHz)':<14} | {'H_n0':<14} | {'epsilon_p':<12} | {'Aliased':<8}")
    print("-" * 68)
    for o, f0, h0, ep, al in zip(orders_arr, freqs_l0, heights_l0, eps_p, alias_flags):
        print(f"{o:<10} | {f0:<14.4f} | {h0:<14.4e} | {ep:<12.4f} | {str(al):<8}")

    return all_measured_modes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SPANDA-Stage 2b: radial-mode frequencies and heights for one star.")
    parser.add_argument('kic', type=int, help="KIC number")
    parser.add_argument('--stage1-csv', required=True, help="Stage 1 summary CSV (stage1_summary_<N>.csv)")
    parser.add_argument('--psd-dir', required=True, help="folder containing the power spectra")
    parser.add_argument('--psd-pattern', default="{kic}.npy", help="PSD file name with {kic} placeholder (default: {kic}.npy)")
    parser.add_argument('--epsilon-summary', default="./stage2a_epsilon_p_summary.txt",
                        help="table written by stage2_collect_epsilon_p.py")
    parser.add_argument('--outdir', default="./p_mode_frequencies", help="folder for <KIC>_p_frequencies.npy")
    parser.add_argument('--plot', action='store_true', help="save diagnostic plots")
    parser.add_argument('--plot-dir', default="./stage2b_plots", help="folder for the diagnostic plots")
    parser.add_argument('--alias', action='store_true', help="also search for Nyquist-aliased modes")
    parser.add_argument('--no-file', action='store_true', help="do not save the .npy output")
    args = parser.parse_args()

    main(kic=args.kic, stage1_csv=args.stage1_csv, epsilon_summary=args.epsilon_summary,
         psd_dir=args.psd_dir, psd_pattern=args.psd_pattern, outdir=args.outdir, plot_dir=args.plot_dir,
         gen_plot=args.plot, gen_alias=args.alias, gen_file=not args.no_file)
