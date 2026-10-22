#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 3a: radial-mode (l=0) fit (Differential Evolution + HMC)
# Fit results are printed to stdout (redirect them to a log file).
# Author: Shatanik Bhattacharya, Co-contributor: Jharnesh Verma
# Usage:  python3 stage3a_l0_fit.py <KIC> <STAGE1_CSV> <P_FREQ_DIR> --psd-dir <DIR>
# ============================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter
from scipy.optimize import differential_evolution
import corner

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from numpyro.diagnostics import summary

CORNER_TITLE_FONTSIZE = 17
CORNER_LABEL_FONTSIZE = 20
CORNER_TICK_FONTSIZE = 13
CORNER_XLABELPAD = 0.35
CORNER_YLABELPAD = 0.70

PLOT_TITLE_FONTSIZE = 18
PLOT_AXIS_LABEL_FONTSIZE = 18
PLOT_TICK_FONTSIZE = 16
PLOT_LEGEND_FONTSIZE = 15
ISLAND_TITLE_FONTSIZE = 14
ISLAND_LEGEND_FONTSIZE = 12
RESIDUAL_BOX_FONTSIZE = 16

LOGPOST_CHUNK = 256


# ============================================================
# 1. Models
# ============================================================

def jax_bkg_fixed(x, fixed_bkg_params, nyq):
    """Fixed MAP background (without the Gaussian envelope)."""
    _, log10_a1_b1, log10_b1, log10_a2, log10_a3, log10_b2, log10_b3, log10_W, _, _ = fixed_bkg_params

    a1_b1, b1 = 10**log10_a1_b1, 10**log10_b1
    a2, a3 = 10**log10_a2, 10**log10_a3
    b2, b3 = 10**log10_b2, 10**log10_b3
    W = 10**log10_W

    t1 = a1_b1 / (1 + (x / b1)**2)
    t2 = (a2 / b2) / (1 + (x / b2)**4)
    t3 = (a3 / b3) / (1 + (x / b3)**4)

    sinc_sq = (jnp.sinc(0.5 * x / nyq))**2
    return t1 + sinc_sq * (t2 + t3) + W


@jax.jit
def jax_modes(x, H0, nu0, gamma0, nyq, delta_f):
    """K l=0 profiles: Lorentzian if gamma0 >= 2*delta_f/pi, sinc^2 otherwise."""
    x_grid = x[:, None]
    H0_grid = H0[None, :]
    nu0_grid = nu0[None, :]

    lorentz = H0_grid / (1.0 + (2.0 * (x_grid - nu0_grid) / gamma0)**2)
    unresolved = H0_grid * (jnp.sinc((x_grid - nu0_grid) / delta_f))**2

    is_resolved = gamma0 >= (2.0 * delta_f / jnp.pi)
    profile = jnp.where(is_resolved, lorentz, unresolved)
    sum_modes = jnp.sum(profile, axis=1)

    return (jnp.sinc(0.5 * x / nyq))**2 * sum_modes


# ============================================================
# 2. Input
# ============================================================

def load_psd(psd_dir, pattern, kic):
    path = Path(psd_dir) / pattern.format(kic=kic)
    if not path.is_file():
        raise FileNotFoundError(f"PSD file not found: {path}")
    arr = np.load(path)
    if arr.shape[0] < arr.shape[1]:
        return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float), path
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float), path


def load_best_bkg_params(KIC, csv_filepath):
    """MAP background parameters and dnu_input from the Stage 1 summary CSV."""
    df = pd.read_csv(csv_filepath, low_memory=False)
    df.columns = df.columns.str.strip()

    kic_data = df[df["kic"] == int(KIC)]
    if kic_data.empty:
        raise ValueError(f"KIC {KIC} not found in {csv_filepath}")

    target_cols = [
        "nu_rmmax_map", "log_10_a_1_over_b_1_map", "log_10_b_1_map", "log_10_a_2_map",
        "log_10_a_3_map", "log_10_b_2_map", "log_10_b_3_map", "log_10_W_map",
        "log_10_H_env_map", "HWHM_map",
    ]
    params_array = kic_data[target_cols].values.flatten().astype(np.float64)
    dnu_input = float(kic_data["dnu_input"].values[0])
    return params_array, dnu_input


def load_rad_details_and_dnu(KIC, fitted_numax, fitted_HWHM, dnu_input, pfreq_dir):
    """Stage 2 mode guesses, local Dnu, and the modes within the fitting range."""
    filepath = Path(pfreq_dir) / f"{KIC}_p_frequencies.npy"
    if not filepath.is_file():
        raise FileNotFoundError(f"Stage 2 l=0 mode file not found: {filepath}")
    print(f"Loading l=0 mode frequencies from: {filepath}")
    data_dict = np.load(filepath, allow_pickle=True).item()

    nu0_all = np.asarray(data_dict['nu_l0'], dtype=np.float64).flatten()
    H0_all = np.asarray(data_dict['H_n0'], dtype=np.float64).flatten()

    idx_closest = np.argsort(np.abs(nu0_all - fitted_numax))[:4]
    closest_nu = np.sort(nu0_all[idx_closest])

    dynamic_dnu = np.mean(np.diff(closest_nu))
    print(f"Dynamically calculated Dnu = {dynamic_dnu:.4f} µHz from closest 4 modes.")

    half_width = 2.0 * max(fitted_HWHM, dnu_input)
    lower_bound = fitted_numax - half_width
    upper_bound = fitted_numax + half_width
    print(f"Filtering modes in range: [{lower_bound:.2f}, {upper_bound:.2f}] µHz")

    mask = (nu0_all >= lower_bound) & (nu0_all <= upper_bound)

    gamma_guess = 0.026 + 0.12 * np.log10(dynamic_dnu)
    d02_guess = 0.121 * dynamic_dnu + 0.047

    return nu0_all[mask], H0_all[mask], dynamic_dnu, gamma_guess, d02_guess


# ============================================================
# 3. Likelihood wrappers
# ============================================================

def build_de_loss_batch(K, delta_f):
    @jax.jit
    def de_loss_batch(thetas, x_fit, y_fit, y_bkg_fixed_fit, nyq):
        def single_loss(theta):
            gamma0 = theta[0]
            log10_H0 = theta[1:K+1]
            nu0 = theta[K+1:2*K+1]
            H0 = 10.0 ** log10_H0
            y_model = y_bkg_fixed_fit + jax_modes(x_fit, H0, nu0, gamma0, nyq, delta_f)
            nll = jnp.sum(jnp.log(y_model) + y_fit / y_model)
            invalid = jnp.any((y_model <= 0) | (~jnp.isfinite(y_model)))
            return jnp.where(invalid, 1e12, nll)
        return jax.vmap(single_loss)(thetas)
    return de_loss_batch


def neg_log_probability_vectorized(theta_population, loss_func, x_fit, y_fit, y_bkg_fixed_fit, bounds, nyq):
    theta_arr = np.asarray(theta_population)
    is_1d = theta_arr.ndim == 1
    thetas = theta_arr[np.newaxis, :] if is_1d else theta_arr.T

    n_candidates = thetas.shape[0]
    low_bounds = np.array([b[0] for b in bounds])
    high_bounds = np.array([b[1] for b in bounds])

    out_of_bounds = np.any((thetas <= low_bounds) | (thetas >= high_bounds), axis=-1)
    results = np.full(n_candidates, 1e12, dtype=np.float64)

    in_bounds_mask = ~out_of_bounds
    if np.any(in_bounds_mask):
        valid_thetas = jnp.asarray(thetas[in_bounds_mask])
        computed_nll = loss_func(valid_thetas, x_fit, y_fit, y_bkg_fixed_fit, nyq)
        results[in_bounds_mask] = np.asarray(computed_nll)

    return results[0] if is_1d else results


def numpyro_model(x_fit, y_bkg_fixed_fit, nyq, K, bounds_dict, delta_f, y_obs=None):
    gamma0 = numpyro.sample("gamma0", dist.Uniform(bounds_dict['gamma'][0], bounds_dict['gamma'][1]))

    log10_H0_low = jnp.array([b[0] for b in bounds_dict['log10_H0']])
    log10_H0_high = jnp.array([b[1] for b in bounds_dict['log10_H0']])
    log10_H0 = numpyro.sample("log10_H0", dist.Uniform(log10_H0_low, log10_H0_high))
    H0 = 10.0 ** log10_H0

    nu0_low = jnp.array([b[0] for b in bounds_dict['nu0']])
    nu0_high = jnp.array([b[1] for b in bounds_dict['nu0']])
    nu0 = numpyro.sample("nu0", dist.Uniform(nu0_low, nu0_high))

    y_model = y_bkg_fixed_fit + jax_modes(x_fit, H0, nu0, gamma0, nyq, delta_f)
    numpyro.sample("obs", dist.Exponential(rate=1.0 / y_model), obs=y_obs)


def compute_log_posterior_samples(samples, model_kwargs, chunk=LOGPOST_CHUNK):
    """Log-posterior (log prior + log-likelihood) of every sample, in the constrained space."""
    def lp_single(params):
        lp, _ = log_density(numpyro_model, (), model_kwargs, params)
        return lp

    lp_batch = jax.jit(jax.vmap(lp_single))
    n = int(samples["gamma0"].shape[0])
    out = []
    for start in range(0, n, chunk):
        sub = {k: v[start:start + chunk] for k, v in samples.items()}
        out.append(np.asarray(lp_batch(sub)))
    return np.concatenate(out)


# ============================================================
# 4. Main routine
# ============================================================

def main(KIC_number, csv_filepath, pfreq_dir, psd_dir, psd_pattern, outdir):
    KIC = int(float(KIC_number))

    if not Path(csv_filepath).is_file():
        raise FileNotFoundError(f"Stage 1 CSV not found: {csv_filepath}")
    if not Path(pfreq_dir).is_dir():
        raise FileNotFoundError(f"Stage 2 p-frequency folder not found: {pfreq_dir}")

    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    pdf_filename = folder / f"l0_fit_DE_HMC_{KIC}.pdf"

    print("==================================================")
    print(f" KIC {KIC} - l=0 Mode Fitting")
    print(f" Stage 1 CSV       : {csv_filepath}")
    print(f" Stage 2 modes     : {Path(pfreq_dir) / f'{KIC}_p_frequencies.npy'}")
    print(f" Output PDF        : {pdf_filename}")
    print("==================================================")
    net_start = time.perf_counter()

    x_bkg, y_bkg_obs, _ = load_psd(psd_dir, psd_pattern, KIC)
    nyq = x_bkg[-1]
    delta_f = float(np.median(np.diff(x_bkg)))
    resolved_threshold = 2.0 * delta_f / np.pi
    print(f"Frequency resolution delta_f = {delta_f:.6f} µHz "
          f"(resolved/unresolved threshold on Gamma0: {resolved_threshold:.6f} µHz)")

    best_bkg_params, dnu_input = load_best_bkg_params(KIC, csv_filepath)
    fitted_numax = best_bkg_params[0]
    fitted_HWHM = best_bkg_params[9]
    y_bkg_fixed_full = np.array(jax_bkg_fixed(jnp.array(x_bkg), jnp.array(best_bkg_params), nyq))

    nu_guesses, H0_guesses, dnu, gamma_guess, d02_guess = load_rad_details_and_dnu(
        KIC, fitted_numax, fitted_HWHM, dnu_input, pfreq_dir
    )
    filter_half_width = 2.0 * max(fitted_HWHM, dnu_input)
    filter_lower = fitted_numax - filter_half_width
    filter_upper = fitted_numax + filter_half_width
    K = len(nu_guesses)
    D = 2 * K + 1
    print(f"Identified {K} l=0 modes in range. Total parameters: {D}")
    print(f"Computed Gamma_guess: {gamma_guess:.4f} µHz.")

    mode_mask = np.zeros_like(x_bkg, dtype=bool)
    island_left = d02_guess / 3.0
    island_right = d02_guess / 3.0
    for nu_g in nu_guesses:
        mode_mask |= (x_bkg >= nu_g - island_left) & (x_bkg <= nu_g + island_right)

    x_fit = jnp.array(x_bkg[mode_mask])
    y_fit = jnp.array(y_bkg_obs[mode_mask])
    y_bkg_fixed_fit = jnp.array(y_bkg_fixed_full[mode_mask])

    bounds, bounds_dict = [], {'log10_H0': [], 'nu0': []}

    gamma_upper = min(0.25, 4.00 * gamma_guess)
    gamma_lower = min(0.02, 0.25 * gamma_guess)
    bounds.append((gamma_lower, gamma_upper))
    bounds_dict['gamma'] = (gamma_lower, gamma_upper)

    for H_g in H0_guesses:
        low = np.log10(0.01 * H_g)
        high = np.log10(100.0 * H_g)
        bounds.append((low, high))
        bounds_dict['log10_H0'].append((low, high))

    for nu_g in nu_guesses:
        bounds.append((nu_g - gamma_guess, nu_g + gamma_guess))
        bounds_dict['nu0'].append((nu_g - gamma_guess, nu_g + gamma_guess))

    labels = [r"$\Gamma_0$"] + [rf"$\log_{{10}} H_{{0,{i}}}$" for i in range(K)] + [rf"$\nu_{{0,{i}}}$" for i in range(K)]

    # --- Step 1: Differential Evolution ---
    print(f"\n--- Running DE (Popsize=100, Actual population={100*D}) ---")
    de_loss_batch = build_de_loss_batch(K, delta_f)
    t0_de = time.perf_counter()

    de_result = differential_evolution(
        neg_log_probability_vectorized, bounds,
        args=(de_loss_batch, x_fit, y_fit, y_bkg_fixed_fit, bounds, nyq),
        strategy='best1bin', maxiter=3000, popsize=100, tol=0.01,
        mutation=(0.5, 1.0), recombination=0.7, disp=True,
        vectorized=True, updating='deferred'
    )
    best_fit_de = de_result.x
    print(f"DE Finished in {time.perf_counter() - t0_de:.1f} sec. NLL: {de_result.fun:.2f}")

    # --- Step 2: HMC (NUTS) ---
    dyn_warmup = 500 + (50 * D)
    dyn_samples = 1500 + (100 * D)
    print(f"\n--- Running NUTS (Warmup={dyn_warmup}, Samples={dyn_samples}) ---")
    t0_hmc = time.perf_counter()

    init_values = {
        "gamma0": best_fit_de[0],
        "log10_H0": jnp.array(best_fit_de[1:K+1]),
        "nu0": jnp.array(best_fit_de[K+1:])
    }
    init_strategy = numpyro.infer.init_to_value(values=init_values)
    kernel = NUTS(numpyro_model, target_accept_prob=0.90, init_strategy=init_strategy, dense_mass=True)

    model_kwargs = dict(x_fit=x_fit, y_bkg_fixed_fit=y_bkg_fixed_fit, nyq=nyq, K=K,
                        bounds_dict=bounds_dict, delta_f=delta_f, y_obs=y_fit)

    hmc = MCMC(kernel, num_warmup=dyn_warmup, num_samples=dyn_samples, num_chains=1, progress_bar=False)
    nuts_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"NUTS seed: {nuts_seed}")
    rng_key = jax.random.PRNGKey(nuts_seed)
    hmc.run(rng_key, **model_kwargs, extra_fields=("diverging",))

    samples = hmc.get_samples()
    extra_fields = hmc.get_extra_fields()
    jax.block_until_ready((samples, extra_fields))

    print(f"HMC Finished in {time.perf_counter() - t0_hmc:.1f} sec.")

    # --- Step 3: Sampling diagnostics ---
    t0_diag = time.perf_counter()
    print("\n--- Sampling Diagnostics (single chain; split R-hat) ---")
    divergences = extra_fields["diverging"].sum()

    flat_samples = np.column_stack([samples["gamma0"], samples["log10_H0"], samples["nu0"]])

    print("\n--- Sample Health Check ---")
    any_bad = False
    for i, label in enumerate(labels):
        arr = flat_samples[:, i]
        n_nan = int(np.isnan(arr).sum())
        n_unique = len(np.unique(arr))
        flagged = n_nan > 0 or n_unique < dyn_samples // 10
        any_bad = any_bad or flagged
        print(f"  {label:<20} shape={arr.shape} NaNs={n_nan} unique={n_unique}" + ("  <-- CHECK THIS" if flagged else ""))
    if any_bad:
        print("NOTE: at least one parameter has NaNs or fewer unique values than 10% of the samples (marked above).")
    else:
        print("No NaNs; every parameter has unique values for at least 10% of the samples.")

    diag_summary = summary(samples, prob=0.90, group_by_chain=False)
    r_hats = np.concatenate([np.atleast_1d(diag_summary[k]["r_hat"]) for k in ("gamma0", "log10_H0", "nu0")])
    n_effs = np.concatenate([np.atleast_1d(diag_summary[k]["n_eff"]) for k in ("gamma0", "log10_H0", "nu0")])

    max_r_hat = np.max(r_hats)
    min_n_eff = np.min(n_effs)

    print(f"Max split R-hat (single chain) : {max_r_hat:.3f}   (reference value: < 1.01)")
    print(f"Min effective sample size      : {min_n_eff:.1f}   (reference value: > 400)")
    print(f"Divergent transitions          : {int(divergences)}   (reference value: 0)")

    flags = []
    if max_r_hat >= 1.01:
        flags.append("split R-hat >= 1.01")
    if min_n_eff <= 400:
        flags.append("n_eff <= 400")
    if divergences > 0:
        flags.append("divergences > 0")
    print("Outside reference values       : " + (", ".join(flags) if flags else "none"))

    median_hmc_params = np.median(flat_samples, axis=0)

    # MAP: the posterior sample with the highest log-posterior
    t0_lp = time.perf_counter()
    log_post = compute_log_posterior_samples(samples, model_kwargs)
    n_bad_lp = int(np.sum(~np.isfinite(log_post)))
    if n_bad_lp == len(log_post):
        raise RuntimeError("Log-posterior is non-finite for every posterior sample.")
    best_hmc_idx = int(np.nanargmax(np.where(np.isfinite(log_post), log_post, np.nan)))
    best_hmc_map_params = flat_samples[best_hmc_idx]
    print(f"\nMAP sample: index {best_hmc_idx} of {len(log_post)} | "
          f"log-posterior = {log_post[best_hmc_idx]:.4f} "
          f"(evaluated for all samples in {time.perf_counter() - t0_lp:.2f} sec"
          + (f"; {n_bad_lp} non-finite values ignored" if n_bad_lp else "") + ")")

    print("\n--- HMC Inferred Parameters ---")
    print(f"{'Parameter':<25} | {'Median':<9} | {'Uncertainty (+ / -)':<20} | {'MAP sample':<12} | {'In 1-Sigma'}")
    print("-" * 88)
    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        best_val = best_hmc_map_params[i]
        in_1sigma = q16 <= best_val <= q84
        print(f"{label:<25} | {q50:9.4f} | +{q84-q50:.4f} / -{q50-q16:.4f}     | {best_val:9.4f}   | {in_1sigma}")

    print(f"\n[TIMING] Diagnostics calculation took {time.perf_counter() - t0_diag:.2f} sec.")

    best_gamma0 = best_hmc_map_params[0]
    best_H0 = 10.0 ** best_hmc_map_params[1:K+1]
    best_nu0 = best_hmc_map_params[K+1:]

    is_resolved_map = bool(best_gamma0 >= resolved_threshold)
    print(f"Gamma0 (MAP) = {best_gamma0:.5f} µHz -> "
          f"{'RESOLVED (Lorentzian profile)' if is_resolved_map else 'UNRESOLVED (sinc^2 profile)'}")

    y_mode_full = np.array(jax_modes(jnp.array(x_bkg), jnp.array(best_H0), jnp.array(best_nu0), best_gamma0, nyq, delta_f))
    y_model_full = y_bkg_fixed_full + y_mode_full

    median_gamma0 = median_hmc_params[0]
    median_H0 = 10.0 ** median_hmc_params[1:K+1]
    median_nu0 = median_hmc_params[K+1:]
    y_mode_full_median = np.array(jax_modes(jnp.array(x_bkg), jnp.array(median_H0), jnp.array(median_nu0), median_gamma0, nyq, delta_f))
    y_model_full_median = y_bkg_fixed_full + y_mode_full_median

    print("\n--- Generating Output Document ---")
    with PdfPages(pdf_filename) as pdf:
        # 1. Corner plot
        fig_corner = corner.corner(
            flat_samples, labels=labels, truths=best_hmc_map_params, truth_color='gold',
            quantiles=[0.16, 0.5, 0.84], show_titles=True,
            title_kwargs={"fontsize": CORNER_TITLE_FONTSIZE, "ha": "left", "x": 0.0, "pad": 8},
            label_kwargs={"fontsize": CORNER_LABEL_FONTSIZE},
            max_n_ticks=2, figsize=(16, 16)
        )
        ndim = D
        axes_grid = np.array(fig_corner.axes).reshape((ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                ax = axes_grid[i, j]
                if j > i:
                    ax.set_axis_off()
                    ax.cla()
                    if ax in fig_corner.axes:
                        fig_corner.delaxes(ax)
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune=None))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune=None))
                    if i == ndim - 1:
                        formatter_x = ScalarFormatter(useOffset=False)
                        formatter_x.set_scientific(False)
                        ax.xaxis.set_major_formatter(formatter_x)
                        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize=CORNER_TICK_FONTSIZE)
                        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
                        ax.xaxis.set_label_coords(0.5, -CORNER_XLABELPAD)
                    else:
                        ax.xaxis.set_major_formatter(NullFormatter())
                        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
                    if j == 0 and i > 0:
                        formatter_y = ScalarFormatter(useOffset=False)
                        formatter_y.set_scientific(False)
                        ax.yaxis.set_major_formatter(formatter_y)
                        ax.tick_params(axis="y", which="both", left=True, labelleft=True, labelsize=CORNER_TICK_FONTSIZE)
                        plt.setp(ax.get_yticklabels(), rotation=20, ha="right", rotation_mode="anchor")
                        ax.yaxis.set_label_coords(-CORNER_YLABELPAD, 0.5)
                    elif j == 0 and i == 0:
                        ax.yaxis.set_major_formatter(NullFormatter())
                        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                    else:
                        ax.yaxis.set_major_formatter(NullFormatter())
                        ax.tick_params(axis="y", which="both", left=True, labelleft=False)
        for i in range(ndim):
            diag_title = axes_grid[i, i].title
            if diag_title:
                diag_title.set_clip_on(False)
        fig_corner.subplots_adjust(left=0.12, bottom=0.12, right=0.94, top=0.94, hspace=0.25, wspace=0.25)
        pdf.savefig(fig_corner, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_corner)

        # 2. Island panels
        fig_zoom, axes_zoom = plt.subplots(K, 1, figsize=(8, 2.8 * K))
        if K == 1:
            axes_zoom = [axes_zoom]
        for i, ax in enumerate(axes_zoom):
            nu_g = nu_guesses[i]
            z_mask = (x_bkg >= nu_g - 0.2*dnu) & (x_bkg <= nu_g + 0.2*dnu)
            ax.axvspan(nu_g - island_left, nu_g + island_right, color='khaki', alpha=0.35, label='Island window')
            ax.plot(x_bkg[z_mask], y_bkg_obs[z_mask], color='black', alpha=0.55, lw=0.8, label="Observations")
            ax.plot(x_bkg[z_mask], y_model_full[z_mask], color='crimson', lw=2.2, label="Total fit (HMC MAP)")
            ax.plot(x_bkg[z_mask], y_bkg_fixed_full[z_mask], color='steelblue', linestyle='--', lw=1.6, label="Fixed background")
            ax.axvline(best_nu0[i], color='darkgreen', linestyle='-', lw=1.8, label=r"Fitted $\nu_{0}$")
            ax.axvline(nu_g, color='darkorange', linestyle='--', lw=1.8, label="Initial guess")
            ax.set_title(f"Island {i+1} near {nu_g:.2f} µHz", fontsize=ISLAND_TITLE_FONTSIZE)
            ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
            ax.legend(fontsize=ISLAND_LEGEND_FONTSIZE, ncol=2, loc='best')
        plt.tight_layout()
        pdf.savefig(fig_zoom)
        plt.close(fig_zoom)

        # 3. Wide-region comparison
        filt_mask = (x_bkg >= filter_lower) & (x_bkg <= filter_upper)
        fig_wide, ax_wide = plt.subplots(figsize=(11, 6))
        ax_wide.plot(x_bkg[filt_mask], y_bkg_obs[filt_mask], color='black', alpha=0.35, lw=0.6, label="Observations")
        ax_wide.plot(x_bkg[filt_mask], y_bkg_fixed_full[filt_mask], color='steelblue', linestyle='--', lw=1.8, label="Fixed background (MAP)")
        ax_wide.plot(x_bkg[filt_mask], y_model_full[filt_mask], color='crimson', lw=2.0, label="Total fit (l=0 MAP + background)")
        ax_wide.plot(x_bkg[filt_mask], y_model_full_median[filt_mask], color='teal', linestyle='-.', lw=2.0, label="Total fit (l=0 median + background)")
        for nu_g in nu_guesses:
            ax_wide.axvspan(nu_g - island_left, nu_g + island_right, color='khaki', alpha=0.3)
        ax_wide.set_xlim(filter_lower, filter_upper)
        ax_wide.set_xlabel(r"Frequency ($\mu$Hz)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_wide.set_ylabel(r"PSD (ppm$^2/\mu$Hz)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_wide.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
        ax_wide.legend(loc='best', fontsize=PLOT_LEGEND_FONTSIZE, ncol=2)
        ax_wide.set_ylim(0, 1.2*np.max(y_bkg_obs[filt_mask]))
        plt.tight_layout()
        pdf.savefig(fig_wide)
        plt.close(fig_wide)

        # 4. Fractional residuals in the islands
        y_model_fit = np.array(y_bkg_fixed_fit + jax_modes(x_fit, jnp.array(best_H0), jnp.array(best_nu0), best_gamma0, nyq, delta_f))
        frac_res = np.array(y_fit / y_model_fit)

        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        ax_res.hist(frac_res, bins=50, density=True, color='gold', alpha=0.7, label='Fractional Residuals')
        x_vals = np.linspace(0, np.max(frac_res), 200)
        ax_res.plot(x_vals, np.exp(-x_vals), 'r--', lw=2, label=r'$e^{-x}$')
        ax_res.set_yscale('log')
        ax_res.set_ylim(bottom=1e-4, top=2)
        ax_res.set_xlabel("Fractional Residuals (Observed / Modelled)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_res.set_ylabel("Density", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_res.set_title("Normalized Distribution of Fractional Residuals in Islands", fontsize=PLOT_TITLE_FONTSIZE)
        ax_res.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
        ax_res.text(0.70, 0.95, f"Mean = {np.mean(frac_res):.4f}\nStd Dev = {np.std(frac_res):.4f}",
                    transform=ax_res.transAxes, fontsize=RESIDUAL_BOX_FONTSIZE, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_res.legend(loc='upper right', bbox_to_anchor=(0.98, 0.80), fontsize=PLOT_LEGEND_FONTSIZE)
        plt.tight_layout()
        pdf.savefig(fig_res)
        plt.close(fig_res)

    print(f"Output PDF written to: {pdf_filename}")
    print(f"\nTotal Pipeline Time: {(time.perf_counter() - net_start)/60:.2f} minutes")
    print("==================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SPANDA-Stage 3a: radial-mode (l=0) fit (DE + HMC) for one star.")
    ap.add_argument("kic", help="KIC number")
    ap.add_argument("stage1_csv", help="Stage 1 summary CSV (stage1_summary_<N>.csv)")
    ap.add_argument("pfreq_dir", help="Stage 2 folder with <KIC>_p_frequencies.npy")
    ap.add_argument("--psd-dir", required=True, help="folder containing the power spectra")
    ap.add_argument("--psd-pattern", default="{kic}.npy", help="PSD file name with {kic} placeholder (default: {kic}.npy)")
    ap.add_argument("--outdir", default="./stage3a_fit_pdfs", help="folder for the PDF (default: ./stage3a_fit_pdfs)")
    args = ap.parse_args()
    main(args.kic, args.stage1_csv, args.pfreq_dir, args.psd_dir, args.psd_pattern, args.outdir)
