#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 1: background fit (Differential Evolution + HMC)
# Output: multi-page PDF with corner plot, fit, residuals and traces;
#         fit results are printed to stdout (redirect them to a log file).
# Author: Shatanik Bhattacharya, Co-contributor: Jharnesh Verma
# Usage:  python3 stage1_background_fit.py <KIC> --catalog <CSV> --psd-dir <DIR>
# ============================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import csv
import time
from pathlib import Path

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
from numpyro.diagnostics import summary


# ============================================================
# 1. Background model and likelihood
# ============================================================

def jax_generate_bkg(x, theta, nyq, numax):
    numax_val, log10_a1_b1, log10_b1, log10_a2, log10_a3, log10_b2, log10_b3, log10_W, log10_Henv, lin_HWHM = theta

    a1_b1 = 10**log10_a1_b1
    b1    = 10**log10_b1
    a2    = 10**log10_a2
    a3    = 10**log10_a3
    b2    = 10**log10_b2
    b3    = 10**log10_b3
    W     = 10**log10_W

    Henv  = 10**log10_Henv
    HWHM  = lin_HWHM
    sigma = HWHM / jnp.sqrt(2 * jnp.log(2))

    t1 = a1_b1 / (1 + (x / b1)**2)
    t2 = (a2 / b2) / (1 + (x / b2)**4)
    t3 = (a3 / b3) / (1 + (x / b3)**4)

    gauss = Henv * jnp.exp(-0.5 * ((x - numax_val) / sigma)**2)
    sinc_sq = (jnp.sinc(0.5 * x / nyq))**2

    y_model = t1 + sinc_sq * (t2 + t3 + gauss) + W
    return y_model


@jax.jit
def _neg_log_prob_batch(thetas, x, y_obs, nyq, numax):
    batched_bkg = jax.vmap(jax_generate_bkg, in_axes=(None, 0, None, None))
    y_models = batched_bkg(x, thetas, nyq, numax)
    nll = jnp.sum(jnp.log(y_models) + y_obs / y_models, axis=-1)
    invalid = jnp.any((y_models <= 0) | (~jnp.isfinite(y_models)), axis=-1)
    return jnp.where(invalid, 1e12, nll)


def neg_log_probability_vectorized(theta_population, x, y_obs, bounds, nyq, numax):
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
        computed_nll = _neg_log_prob_batch(valid_thetas, jnp.asarray(x), jnp.asarray(y_obs), nyq, numax)
        results[in_bounds_mask] = np.asarray(computed_nll)

    return results[0] if is_1d else results


def np_neg_log_likelihood(theta, x, y_obs, nyq):
    single = np.ndim(theta) == 1
    thetas = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    (numax_val, log10_a1_b1, log10_b1, log10_a2, log10_a3,
     log10_b2, log10_b3, log10_W, log10_Henv, lin_HWHM) = [thetas[:, i:i + 1] for i in range(thetas.shape[1])]

    a1_b1 = 10**log10_a1_b1
    b1    = 10**log10_b1
    a2    = 10**log10_a2
    a3    = 10**log10_a3
    b2    = 10**log10_b2
    b3    = 10**log10_b3
    W     = 10**log10_W

    Henv  = 10**log10_Henv
    sigma = lin_HWHM / np.sqrt(2 * np.log(2))

    t1 = a1_b1 / (1 + (x / b1)**2)
    t2 = (a2 / b2) / (1 + (x / b2)**4)
    t3 = (a3 / b3) / (1 + (x / b3)**4)

    gauss = Henv * np.exp(-0.5 * ((x - numax_val) / sigma)**2)
    sinc_sq = (np.sinc(0.5 * x / nyq))**2

    y_model = t1 + sinc_sq * (t2 + t3 + gauss) + W
    with np.errstate(divide='ignore', invalid='ignore'):
        nll = np.sum(np.log(y_model) + y_obs / y_model, axis=-1)
    invalid = np.any((y_model <= 0) | (~np.isfinite(y_model)), axis=-1)
    nll = np.where(invalid, 1e12, nll)
    return nll[0] if single else nll


# ============================================================
# 2. Probabilistic model
# ============================================================

def numpyro_model(x, nyq, numax, bounds, y_obs=None):
    numax_val   = numpyro.sample("numax_val",   dist.Uniform(bounds[0][0], bounds[0][1]))
    log10_a1_b1 = numpyro.sample("log10_a1_b1", dist.Uniform(bounds[1][0], bounds[1][1]))
    log10_b1    = numpyro.sample("log10_b1",    dist.Uniform(bounds[2][0], bounds[2][1]))
    log10_a2    = numpyro.sample("log10_a2",    dist.Uniform(bounds[3][0], bounds[3][1]))
    log10_a3    = numpyro.sample("log10_a3",    dist.Uniform(bounds[4][0], bounds[4][1]))
    log10_b2    = numpyro.sample("log10_b2",    dist.Uniform(bounds[5][0], bounds[5][1]))
    log10_b3    = numpyro.sample("log10_b3",    dist.Uniform(bounds[6][0], bounds[6][1]))
    log10_W     = numpyro.sample("log10_W",     dist.Uniform(bounds[7][0], bounds[7][1]))
    log10_Henv  = numpyro.sample("log10_Henv",  dist.Uniform(bounds[8][0], bounds[8][1]))
    lin_HWHM    = numpyro.sample("lin_HWHM",    dist.Uniform(bounds[9][0], bounds[9][1]))

    theta = jnp.stack([numax_val, log10_a1_b1, log10_b1, log10_a2, log10_a3, log10_b2, log10_b3, log10_W, log10_Henv, lin_HWHM])
    y_model = jax_generate_bkg(x, theta, nyq, numax)
    numpyro.sample("obs", dist.Exponential(rate=1.0 / y_model), obs=y_obs)


# ============================================================
# 3. Input
# ============================================================

def load_psd(psd_dir, pattern, kic):
    """Frequency (muHz) and PSD (ppm^2/muHz) from <psd_dir>/<pattern>; array of shape (N, 2) or (2, N)."""
    path = Path(psd_dir) / pattern.format(kic=kic)
    if not path.is_file():
        raise FileNotFoundError(f"PSD file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or min(arr.shape) < 2:
        raise ValueError(f"PSD array must be 2-D with frequency and PSD as its first two columns; got {arr.shape}")
    if arr.shape[0] < arr.shape[1]:
        return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float)
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)


# ============================================================
# 4. Main routine
# ============================================================

def main(KIC_number, catalog_file, psd_dir, psd_pattern, outdir):
    KIC = int(float(KIC_number))
    print("==================================================")
    print(f"       KIC {KIC} - Background Fitting (DE+HMC)    ")
    print("==================================================")

    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    pdf_filename = folder / f"bkg_fit_DE_HMC_{KIC}.pdf"
    print("Save location:", pdf_filename)

    numax, dnu = None, None
    with open(catalog_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row['KIC'])) == KIC:
                numax = float(row['numax'])
                dnu = float(row['Delnu'])
                break

    if numax is None:
        raise ValueError(f"KIC {KIC} not found in {catalog_file}.")

    print(f"KIC={KIC} | numax={numax:.2f} µHz | Dnu={dnu:.2f} µHz")

    x_bkg, y_bkg_obs = load_psd(psd_dir, psd_pattern, KIC)
    nyq = x_bkg[-1]

    exclusion_lower = numax - 3.5 * dnu
    exclusion_upper = numax + 3.5 * dnu
    mode_mask = (x_bkg >= exclusion_lower) & (x_bkg <= exclusion_upper)
    H_max = np.max(y_bkg_obs[mode_mask]) if np.sum(mode_mask) > 0 else np.max(y_bkg_obs)

    x_fit = x_bkg
    y_fit_obs = y_bkg_obs
    print(f"Maximum of power spectrum in frequency range [{exclusion_lower:.2f} , {exclusion_upper:.2f}] µHz being used for reference.")

    net_start_time = time.perf_counter()

    numax_low, numax_high = numax - 2 * dnu, numax + 2 * dnu
    true_a2_low  = 2*np.sqrt(2)/np.pi * (3382 * numax**-0.609 * 0.5)**2
    true_a2_high = 2*np.sqrt(2)/np.pi * (3382 * numax**-0.609 * 1.5)**2
    true_a3_low  = 2*np.sqrt(2)/np.pi * (3382 * numax**-0.609 * 0.5)**2
    true_a3_high = 2*np.sqrt(2)/np.pi * (3382 * numax**-0.609 * 1.5)**2
    b2low, b2high = 0.317 * numax**0.970 * 0.6, 0.317 * numax**0.970 * 1.4
    b3low, b3high = 0.948 * numax**0.992 * 0.6, 0.948 * numax**0.992 * 2.0
    Henv_low, Henv_high = 0.01 * H_max, 0.5 * H_max
    HWHM_low, HWHM_high = 0.5 * dnu, 4.0 * dnu

    bounds = [
        (numax_low, numax_high),
        (-1.0, 10.0),
        (-1.0, 1.0),
        (np.log10(true_a2_low), np.log10(true_a2_high)),
        (np.log10(true_a3_low), np.log10(true_a3_high)),
        (np.log10(b2low), np.log10(b2high)), (np.log10(b3low), np.log10(b3high)),
        (-0.5, 5),
        (np.log10(Henv_low), np.log10(Henv_high)), (HWHM_low, HWHM_high)
    ]

    labels = [r"$\nu_{\rm{max}}$", r"$\log_{10} (a_1/b_1)$", r"$\log_{10} b_1$", r"$\log_{10} a_2$",
              r"$\log_{10} a_3$", r"$\log_{10} b_2$", r"$\log_{10} b_3$", r"$\log_{10} W$",
              r"$\log_{10} H_{env}$", r"$\mathrm{HWHM}$"]
    param_names = ["numax_val", "log10_a1_b1", "log10_b1", "log10_a2", "log10_a3", "log10_b2", "log10_b3", "log10_W", "log10_Henv", "lin_HWHM"]

    # --- Step 1: Differential Evolution ---
    print("\n--- Running Differential Evolution (Global Search) ---")
    t0_de = time.perf_counter()
    de_result = differential_evolution(
        neg_log_probability_vectorized,
        bounds,
        args=(x_fit, y_fit_obs, bounds, nyq, numax),
        strategy='best1bin',
        maxiter=4200,
        popsize=100,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        vectorized=True,
        updating='deferred'
    )
    best_fit_de = de_result.x
    print(f"DE Finished in {time.perf_counter() - t0_de:.1f} sec. Peak NLL: {de_result.fun:.2f}")

    # --- Step 2: HMC (NUTS) ---
    print("\n--- Running NUTS (HMC) ---")
    t0_hmc = time.perf_counter()

    init_strategy = numpyro.infer.init_to_value(
        values={name: val for name, val in zip(param_names, best_fit_de)}
    )
    kernel = NUTS(numpyro_model, target_accept_prob=0.85, init_strategy=init_strategy, dense_mass=True)

    num_warmup = 1000
    num_samples = 3000
    hmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)

    nuts_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"NUTS seed: {nuts_seed}")
    rng_key = jax.random.PRNGKey(nuts_seed)
    hmc.run(rng_key,
            x=jnp.array(x_fit),
            nyq=nyq,
            numax=numax,
            bounds=bounds,
            y_obs=jnp.array(y_fit_obs),
            extra_fields=("potential_energy", "diverging"))

    samples = hmc.get_samples()
    extra_fields = hmc.get_extra_fields()
    jax.block_until_ready((samples, extra_fields))

    print(f"HMC Finished in {time.perf_counter() - t0_hmc:.1f} sec.")

    # --- Step 3: Convergence diagnostics ---
    t0_diag = time.perf_counter()
    print("\n--- HMC Convergence Diagnostics (single chain) ---")
    divergences = extra_fields["diverging"].sum()

    diag_summary = summary(samples, prob=0.90, group_by_chain=False)
    r_hats = [diag_summary[p]['r_hat'] for p in param_names]
    n_effs = [diag_summary[p]['n_eff'] for p in param_names]

    max_r_hat = np.max(r_hats)
    min_n_eff = np.min(n_effs)

    print(f"Max split R-hat (single chain) : {max_r_hat:.3f} (Criterion: < 1.05, Ideally: < 1.01)")
    print(f"Min Effective Samples  : {min_n_eff:.1f} (Ideal: > 400)")
    print(f"Divergent Transitions  : {divergences} (Ideal: 0)")

    if max_r_hat > 1.05 or divergences > 0:
        print("WARNING: Theoretical convergence criteria not fully met.")
    else:
        print("SUCCESS: HMC chain has converged.")

    flat_samples = np.column_stack([samples[name] for name in param_names])
    median_hmc_params = np.median(flat_samples, axis=0)

    # MAP: the posterior sample with the highest likelihood (uniform priors)
    map_chunk = 100
    nll_samples = np.concatenate([
        np_neg_log_likelihood(flat_samples[i:i + map_chunk], x_fit, y_fit_obs, nyq)
        for i in range(0, len(flat_samples), map_chunk)
    ])
    map_idx = int(np.argmin(nll_samples))
    map_params = flat_samples[map_idx]

    print(f"\nMAP sample: index {map_idx} of {len(flat_samples)} | NLL: {nll_samples[map_idx]:.3f}")

    print("\n--- HMC Inferred Parameters ---")
    print(f"{'Parameter':<25} | {'Median':<9} | {'Uncertainty (+ / -)':<20} | {'MAP':<12} | {'In 1-Sigma'}")
    print("-" * 88)
    for i in range(len(bounds)):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        best_val = map_params[i]
        in_1sigma = q16 <= best_val <= q84
        print(f"{labels[i]:<25} | {q50:9.4f} | +{q84-q50:.4f} / -{q50-q16:.4f}     | {best_val:9.4f}   | {in_1sigma}")

    diag_elapsed = time.perf_counter() - t0_diag
    print(f"\n[TIMING] Diagnostics calculation took {diag_elapsed:.2f} sec.")

    # Power-excess ("inside") region: MAP numax +/- 3.5 MAP HWHM, clipped to the data range
    fitted_numax = map_params[param_names.index("numax_val")]
    fitted_HWHM = map_params[param_names.index("lin_HWHM")]
    inside_lower = max(x_bkg[0], fitted_numax - 3.5 * fitted_HWHM)
    inside_upper = min(x_bkg[-1], fitted_numax + 3.5 * fitted_HWHM)
    mode_mask_fit = (x_bkg >= inside_lower) & (x_bkg <= inside_upper)
    print(f"Fitted 'inside' region: {inside_lower:.2f} to {inside_upper:.2f} µHz "
          f"(numax_fit={fitted_numax:.2f}, HWHM_fit={fitted_HWHM:.3f})")

    y_bkg_fit_median_full = np.array(jax_generate_bkg(jnp.array(x_bkg), jnp.array(median_hmc_params), nyq, numax))
    y_bkg_fit_map_full = np.array(jax_generate_bkg(jnp.array(x_bkg), jnp.array(map_params), nyq, numax))

    residuals = y_bkg_obs / y_bkg_fit_map_full
    residuals_outside = y_bkg_obs[~mode_mask_fit] / y_bkg_fit_map_full[~mode_mask_fit]
    residuals_inside = y_bkg_obs[mode_mask_fit] / y_bkg_fit_map_full[mode_mask_fit]

    res_mean, res_std = np.mean(residuals), np.std(residuals)
    res_mean_outside, res_std_outside = np.mean(residuals_outside), np.std(residuals_outside)
    res_mean_inside, res_std_inside = np.mean(residuals_inside), np.std(residuals_inside)

    print(f"\nMean of residuals (MAP model on full region): {res_mean:.5f} | Std dev: {res_std:.5f}")
    print(f"Mean of residuals (Outside): {res_mean_outside:.5f} | Std dev: {res_std_outside:.5f}")
    print(f"Mean of residuals (Inside): {res_mean_inside:.5f} | Std dev: {res_std_inside:.5f}")

    # ============================================================
    # PDF output
    # ============================================================
    print("\n--- Generating PDF Document ---")
    with PdfPages(pdf_filename) as pdf:

        # 1. Corner plot
        CORNER_TITLE_FONTSIZE = 17
        CORNER_LABEL_FONTSIZE = 20
        CORNER_TICK_FONTSIZE = 13
        CORNER_XLABELPAD = 0.35
        CORNER_YLABELPAD = 0.70

        fig_corner = corner.corner(
            flat_samples, labels=labels,
            truths=map_params, truth_color='gold',
            quantiles=[0.16, 0.5, 0.84], show_titles=True,
            title_kwargs={"fontsize": CORNER_TITLE_FONTSIZE, "ha": "left", "x": 0.0, "pad": 8},
            label_kwargs={"fontsize": CORNER_LABEL_FONTSIZE},
            max_n_ticks=2,
            figsize=(16, 16)
        )

        ndim = len(param_names)
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

        margin_left = 0.12
        margin_bottom = 0.12
        grid_span = 0.82
        fig_corner.subplots_adjust(
            left=margin_left, bottom=margin_bottom,
            right=margin_left + grid_span, top=margin_bottom + grid_span,
            hspace=0.25, wspace=0.25
        )
        pdf.savefig(fig_corner, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_corner)

        # 2. PSD fit plots
        with plt.rc_context({
            'font.size': 12, 'axes.titlesize': 20, 'axes.labelsize': 18,
            'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
        }):
            plot_configs = [
                {'xscale': 'log', 'yscale': 'log'},
                {'xscale': 'linear', 'yscale': 'linear'},
                {'xscale': 'linear', 'yscale': 'log'}
            ]

            numax_val, log10_a1_b1, log10_b1, log10_a2, log10_a3, log10_b2, log10_b3, log10_W, log10_Henv, lin_HWHM = map_params
            a1_b1, b1, a2, a3, b2, b3, W = 10**log10_a1_b1, 10**log10_b1, 10**log10_a2, 10**log10_a3, 10**log10_b2, 10**log10_b3, 10**log10_W
            Henv, HWHM = 10**log10_Henv, lin_HWHM
            sigma_plot = HWHM / np.sqrt(2 * np.log(2))
            sinc_sq_full = (np.sinc(0.5 * x_bkg / nyq))**2

            comp_t1 = a1_b1 / (1 + (x_bkg / b1)**2)
            comp_t2 = sinc_sq_full * (a2 / b2) / (1 + (x_bkg / b2)**4)
            comp_t3 = sinc_sq_full * (a3 / b3) / (1 + (x_bkg / b3)**4)
            comp_gauss = sinc_sq_full * Henv * np.exp(-0.5 * ((x_bkg - numax_val) / sigma_plot)**2)

            for config in plot_configs:
                fig_psd, ax = plt.subplots(figsize=(10, 6))

                ax.plot(x_bkg, y_bkg_obs, 'k', alpha=0.3, lw=0.35, label="Observations")
                ax.plot(x_bkg, np.convolve(y_bkg_obs, np.ones(5)/5, mode='same'), 'lime', lw=0.25, label="5-bin Smoothened")
                ax.axvspan(inside_lower, inside_upper, color='red', alpha=0.2, label=r'Power Excess Region ($\nu_{\rm{max}} \pm 3.5\,\mathrm{HWHM}$)')

                ax.plot(x_bkg, y_bkg_fit_median_full, 'k-', lw=1.5, alpha=0.7, label="Total Fit (HMC Median)")
                ax.plot(x_bkg, y_bkg_fit_map_full, color='gold', linestyle='--', lw=2.0, label="Total Fit (MAP)")

                ax.plot(x_bkg, comp_t1, color='darkorange', linestyle=':', lw=2.5, label='Harvey-like (t1)')
                ax.plot(x_bkg, comp_t2, color='red', linestyle=':', lw=2.5, label='Harvey-like (t2)')
                ax.plot(x_bkg, comp_t3, color='blue', linestyle=':', lw=2.5, label='Harvey-like (t3)')
                ax.plot(x_bkg, comp_gauss, color='cyan', linestyle='-', lw=2.0, label='Gaussian Env')
                ax.axhline(W, color='purple', linestyle=':', lw=2.5, label='White Noise (W)')
                ax.plot(x_bkg, comp_t1 + comp_t2 + comp_t3 + W, color='red', linestyle='-', lw=1.8, label='Background fit (MAP)')

                ax.set_xscale(config['xscale'])
                ax.set_yscale(config['yscale'])
                ax.set_xlim(x_bkg[0], x_bkg[-1])

                if config['yscale'] == 'log':
                    ax.set_ylim(np.min(y_bkg_obs)/2, np.max(y_bkg_obs)*2)
                else:
                    ax.set_ylim(0, np.percentile(y_bkg_obs, 99.5))

                ax.set_xlabel(r"Frequency ($\mu$Hz)")
                ax.set_ylabel(r"PSD (ppm$^2 / \mu$Hz)")
                ax.legend(loc='best', fontsize=12, ncol=2)

                plt.tight_layout()
                pdf.savefig(fig_psd, bbox_inches='tight')
                plt.close(fig_psd)

        # 3. Residual histograms
        def plot_residuals(res_data, r_mean, r_std, title):
            fig = plt.figure(figsize=(8, 6))
            x_vals = np.linspace(np.min(res_data), np.max(res_data), 200)
            y_vals = np.exp(-x_vals)
            plt.hist(res_data, bins=50, density=True, alpha=0.7, color='gold', edgecolor='black', label=title)
            plt.plot(x_vals, y_vals, 'r--', linewidth=2, label=r'Ideal $e^{-x}$')
            plt.ylim(1e-5, 1.3)
            plt.yscale('log')
            plt.text(0.95, 0.95, f"Mean: {r_mean:.5f}\nStd Dev: {r_std:.5f}", transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.xlabel("Residuals (Observed / MAP Model PSD)")
            plt.ylabel("Density (Log Scale)")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title(f"KIC {KIC} {title}")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        plot_residuals(residuals, res_mean, res_std, "Residuals Distribution (Full Region)")
        plot_residuals(residuals_outside, res_mean_outside, res_std_outside, "Residuals Distribution (Outside region)")
        plot_residuals(residuals_inside, res_mean_inside, res_std_inside, "Residuals Distribution (Inside region)")

        # 4. Trace plots
        fig_trace, axes_trace = plt.subplots(len(param_names), 1, figsize=(10, 15), sharex=True)
        for i, (name, ax) in enumerate(zip(param_names, axes_trace)):
            ax.plot(samples[name], color='black', alpha=0.5, lw=0.5)
            ax.set_ylabel(labels[i], fontsize=10)
            if i == len(param_names) - 1:
                ax.set_xlabel('HMC Sample Number')
        fig_trace.suptitle(f"KIC {KIC} HMC Trace Plots (Visual Convergence)", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig_trace, bbox_inches='tight')
        plt.close(fig_trace)

    print(f"\nTotal runtime: {(time.perf_counter() - net_start_time)/60:.2f} minutes")
    print("==================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SPANDA-Stage 1: background fit (DE + HMC) for one star.")
    ap.add_argument("kic", help="KIC number")
    ap.add_argument("--catalog", required=True,
                    help="CSV with columns KIC, numax, Delnu (initial nu_max and Delta nu, in muHz)")
    ap.add_argument("--psd-dir", required=True, help="folder containing the power spectra")
    ap.add_argument("--psd-pattern", default="{kic}.npy",
                    help="PSD file name inside --psd-dir, with {kic} as placeholder (default: {kic}.npy)")
    ap.add_argument("--outdir", default="./stage1_fit_pdfs", help="folder for the PDF (default: ./stage1_fit_pdfs)")
    args = ap.parse_args()
    main(args.kic, args.catalog, args.psd_dir, args.psd_pattern, args.outdir)
