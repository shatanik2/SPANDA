#!/usr/bin/env python3
# ============================================================
# SPANDA-Stage 3b: quadrupole-mode (l=2) fit (Differential Evolution + HMC),
# with the l=0 parameters held fixed at their Stage 3a values (MAP if available).
# Fit results are printed to stdout (redirect them to a log file).
# Author: Shatanik Bhattacharya, Co-contributor: Jharnesh Verma
# Usage:  python3 stage3b_l2_fit.py <KIC> <STAGE1_CSV> <STAGE3A_SUMMARY_CSV> --psd-dir <DIR>
#         <STAGE1_CSV> must be the Stage 1 summary used by the Stage 3a fit of this star.
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
ISLAND_LEGEND_FONTSIZE = 10
RESIDUAL_BOX_FONTSIZE = 16

# Colour-blind-safe palette (Okabe & Ito 2008)
MAP_FIT_COLOR = "#D55E00"
MEDIAN_FIT_COLOR = "#0072B2"
BKG_COLOR = "#CC79A7"
OBS_COLOR = '#555555'
L0_LINE_COLOR = "#000000"
L2_LINE_COLOR = "#56B4E9"
MAP_FIT_LW = 2.0
MEDIAN_FIT_LW = 2.2
MEDIAN_FIT_LS = (0, (5, 2, 1, 2))
BKG_LS = (0, (6, 3))
L2_LINE_LS = (0, (1.5, 1.5))
ISLAND_SHADE_COLOR = "#F0E442"
ISLAND_SHADE_ALPHA = 0.25
RES_HIST_COLOR = "#56B4E9"
RES_CURVE_COLOR = "#D55E00"
CORNER_TRUTH_COLOR = "#E69F00"
Z_SHADE, Z_OBS, Z_BKG, Z_MEDIAN, Z_MAP, Z_VLINE = 0, 1, 2, 3, 4, 5

LOGPOST_CHUNK = 256
FILTER_MATCH_TOL = 0.006
L0_VALUE_TYPES = ("map", "median", "val")


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
def jax_modes(x, H0, nu0, H2, nu2, gamma0, inc_2, nu_rot_2, nyq, delta_f):
    x_grid = x[:, None]
    H0_grid = H0[None, :]
    nu0_grid = nu0[None, :]
    H2_grid = H2[None, :]
    nu2_grid = nu2[None, :]

    is_resolved = gamma0 >= (2.0 * delta_f / jnp.pi)

    lorentz0 = H0_grid / (1.0 + (2.0 * (x_grid - nu0_grid) / gamma0)**2)
    sinc0 = H0_grid * (jnp.sinc((x_grid - nu0_grid) / delta_f))**2
    profile0 = jnp.where(is_resolved, lorentz0, sinc0)

    # l=2 visibility factors E_2m (Gizon & Solanki 2003)
    inc_2_rad = inc_2 * jnp.pi / 180.0
    cos2 = jnp.cos(inc_2_rad)**2
    E20 = 0.25 * (3.0 * cos2 - 1.0)**2
    E21 = 1.5 * cos2 * (1.0 - cos2)
    E22 = 0.375 * (1.0 - cos2)**2

    def split_component(center, E):
        lor = (H2_grid * E) / (1.0 + (2.0 * (x_grid - center) / gamma0)**2)
        sinc_prof = H2_grid * E * (jnp.sinc((x_grid - center) / delta_f))**2
        return jnp.where(is_resolved, lor, sinc_prof)

    L2_0  = split_component(nu2_grid, E20)
    L2_p1 = split_component(nu2_grid - nu_rot_2, E21)
    L2_m1 = split_component(nu2_grid + nu_rot_2, E21)
    L2_p2 = split_component(nu2_grid - 2.0 * nu_rot_2, E22)
    L2_m2 = split_component(nu2_grid + 2.0 * nu_rot_2, E22)

    lorentz2 = L2_0 + L2_p1 + L2_m1 + L2_p2 + L2_m2
    sum_modes = jnp.sum(profile0 + lorentz2, axis=1)

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
        return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float)
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)


def load_best_bkg_params(KIC, csv_filepath):
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


def load_l0_fit_params(KIC, csv_filepath):
    """Fixed l=0 parameters from the Stage 3a summary: Dnu, Gamma_0, nu_0_i, log10 H_0_i
    (one value type for all: MAP if present, else median), and the l=0 mode-filter range."""
    df = pd.read_csv(csv_filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    kic_data = df[df["kic"] == int(KIC)]
    if kic_data.empty:
        raise ValueError(f"KIC {KIC} not found in {csv_filepath}")
    row = kic_data.iloc[0]

    value_type = next((t for t in L0_VALUE_TYPES if f"Gamma_0_{t}" in df.columns), None)
    if value_type is None:
        raise KeyError("No Gamma_0 column (_map, _median or _val) in the Stage 3a summary CSV.")

    dnu_calc = float(row["dnu_calculated"])
    gamma0 = float(row[f"Gamma_0_{value_type}"])
    if not np.isfinite(gamma0):
        raise ValueError(f"KIC {KIC}: Gamma_0_{value_type} is empty in {csv_filepath}")

    for col in ("freq_filter_min", "freq_filter_max"):
        if col not in df.columns:
            raise KeyError(f"Stage 3a summary CSV lacks {col}.")
    l0_filter = (float(row["freq_filter_min"]), float(row["freq_filter_max"]))

    if "num_l0_modes" in df.columns and pd.notna(row["num_l0_modes"]):
        K = int(row["num_l0_modes"])
    else:
        K = 0
        while f"nu_0_{K}_{value_type}" in df.columns and pd.notna(row[f"nu_0_{K}_{value_type}"]):
            K += 1
    if K < 1:
        raise ValueError(f"KIC {KIC}: no fitted l=0 modes in {csv_filepath}")

    nu0_list, log10_H0_list = [], []
    for i in range(K):
        nu_col, h_col = f"nu_0_{i}_{value_type}", f"log_10_H_0_{i}_{value_type}"
        if nu_col not in df.columns or h_col not in df.columns:
            raise KeyError(f"Stage 3a summary CSV lacks {nu_col} / {h_col}")
        nu_val, h_val = row[nu_col], row[h_col]
        if pd.isna(nu_val) or pd.isna(h_val):
            raise ValueError(f"KIC {KIC}: l=0 mode {i} has an empty {value_type} value")
        nu0_list.append(float(nu_val))
        log10_H0_list.append(float(h_val))

    return dnu_calc, gamma0, np.array(log10_H0_list), np.array(nu0_list), value_type, l0_filter


def verify_same_stage1_as_l0(KIC, fitted_numax, fitted_HWHM, dnu_input, l0_filter):
    """Stops unless this Stage 1 CSV reproduces the mode-filter range of the Stage 3a fit."""
    half_width = 2.0 * max(fitted_HWHM, dnu_input)
    lo, hi = fitted_numax - half_width, fitted_numax + half_width
    f_min, f_max = l0_filter
    diff = max(abs(lo - f_min), abs(hi - f_max))
    print(f"Stage 1 check: mode-filter range from this Stage 1 CSV = [{lo:.4f}, {hi:.4f}] µHz; "
          f"Stage 3a summary = [{f_min:.2f}, {f_max:.2f}] µHz; max difference = {diff:.4f} µHz")
    if not diff <= FILTER_MATCH_TOL:
        raise ValueError(f"KIC {KIC}: the Stage 1 CSV does not reproduce the Stage 3a mode-filter range; "
                         f"use the Stage 1 CSV that the Stage 3a fit used.")
    print("Stage 1 check: PASSED (same background as the l=0 fit)")


# ============================================================
# 3. Likelihood wrappers
# ============================================================

def build_de_loss_batch(K, gamma0, H0, nu0, delta_f):
    @jax.jit
    def de_loss_batch(thetas, x_fit, y_fit, y_bkg_fixed_fit, nyq):
        def single_loss(theta):
            inc_2 = theta[0]
            nu_rot_2 = theta[1]
            V2 = theta[2]
            nu2 = theta[3:K+3]
            H2 = V2 * H0
            y_model = y_bkg_fixed_fit + jax_modes(x_fit, H0, nu0, H2, nu2, gamma0, inc_2, nu_rot_2, nyq, delta_f)
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


def numpyro_model(x_fit, y_bkg_fixed_fit, nyq, K, bounds_dict, gamma0, H0, nu0, delta_f, y_obs=None):
    cos_i_2 = numpyro.sample("cos_i_2", dist.Uniform(0.0, 1.0))
    inc_2 = numpyro.deterministic("inc_2", jnp.arccos(cos_i_2) * 180.0 / jnp.pi)

    nu_rot_2 = numpyro.sample("nu_rot_2", dist.Uniform(bounds_dict['nu_rot_2'][0], bounds_dict['nu_rot_2'][1]))
    V2 = numpyro.sample("V2", dist.Uniform(bounds_dict['V2'][0], bounds_dict['V2'][1]))

    nu2 = numpyro.sample("nu2", dist.Uniform(
        jnp.array([b[0] for b in bounds_dict['nu2']]),
        jnp.array([b[1] for b in bounds_dict['nu2']])
    ))

    H2 = V2 * H0
    y_model = y_bkg_fixed_fit + jax_modes(x_fit, H0, nu0, H2, nu2, gamma0, inc_2, nu_rot_2, nyq, delta_f)
    numpyro.sample("obs", dist.Exponential(rate=1.0 / y_model), obs=y_obs)


SAMPLED_SITES = ("cos_i_2", "nu_rot_2", "V2", "nu2")


def compute_log_posterior_samples(samples, model_kwargs, chunk=LOGPOST_CHUNK):
    """Log-posterior of every sample in the constrained space of the sampled sites."""
    def lp_single(params):
        lp, _ = log_density(numpyro_model, (), model_kwargs, params)
        return lp

    lp_batch = jax.jit(jax.vmap(lp_single))
    n = int(samples["cos_i_2"].shape[0])
    out = []
    for start in range(0, n, chunk):
        sub = {k: samples[k][start:start + chunk] for k in SAMPLED_SITES}
        out.append(np.asarray(lp_batch(sub)))
    return np.concatenate(out)


# ============================================================
# 4. Main routine
# ============================================================

def main(KIC_number, bkg_csv, l0_csv, psd_dir, psd_pattern, outdir):
    KIC = int(float(KIC_number))
    if not Path(bkg_csv).is_file():
        raise FileNotFoundError(f"Stage 1 CSV not found: {bkg_csv}")
    if not Path(l0_csv).is_file():
        raise FileNotFoundError(f"Stage 3a summary CSV not found: {l0_csv}")

    print("==================================================")
    print(f" KIC {KIC} - l=2 Mode Fitting (l=0 parameters fixed)")
    print(f" Background CSV Source : {bkg_csv}")
    print(f" l=0 Fit CSV Source    : {l0_csv}")
    print("==================================================")
    net_start = time.perf_counter()

    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    pdf_filename = folder / f"l2_fit_DE_HMC_{KIC}.pdf"

    x_bkg, y_bkg_obs = load_psd(psd_dir, psd_pattern, KIC)
    nyq = x_bkg[-1]
    delta_f = float(np.median(np.diff(x_bkg)))
    resolved_threshold = 2.0 * delta_f / np.pi

    best_bkg_params, dnu_input = load_best_bkg_params(KIC, bkg_csv)
    fitted_numax = best_bkg_params[0]
    fitted_HWHM = best_bkg_params[9]
    y_bkg_fixed_full = np.array(jax_bkg_fixed(jnp.array(x_bkg), jnp.array(best_bkg_params), nyq))
    print(f"Background (Stage 1 MAP): numax = {fitted_numax:.4f} µHz, HWHM = {fitted_HWHM:.4f} µHz, "
          f"dnu_input = {dnu_input:.4f} µHz")

    (dnu, best_gamma0, best_log10_H0_arr, best_nu0_arr,
     l0_value_type, l0_filter) = load_l0_fit_params(KIC, l0_csv)

    verify_same_stage1_as_l0(KIC, fitted_numax, fitted_HWHM, dnu_input, l0_filter)

    best_H0_arr = 10.0 ** best_log10_H0_arr
    K = len(best_nu0_arr)
    D = K + 3
    d02_guess = 0.121 * dnu + 0.047

    print(f"Loaded {K} fixed l=0 modes (l=0 value type used for Gamma0, nu0 and log10 H0: '{l0_value_type}').")
    print(f"Dnu_calculated (from the l=0 fit): {dnu:.4f} µHz.")
    print("Fixed l=0 frequencies (µHz): " + ", ".join(f"{v:.4f}" for v in best_nu0_arr))
    print(f"Fixed Gamma0: {best_gamma0:.4f} µHz, Computed d02_guess: {d02_guess:.4f} µHz.")
    print(f"Frequency resolution delta_f = {delta_f:.6f} µHz "
          f"(resolved/unresolved threshold on Gamma0: {resolved_threshold:.6f} µHz)")
    is_resolved_fixed = bool(best_gamma0 >= resolved_threshold)
    print(f"Fixed Gamma0 status -> {'RESOLVED (Lorentzian profile)' if is_resolved_fixed else 'UNRESOLVED (sinc^2 profile)'}")

    # Fit windows: [nu0_i - 4*d02/3, nu0_i - 2*d02/3] for every fitted l=0 mode
    mode_mask = np.zeros_like(x_bkg, dtype=bool)
    island_left = 4.0 * d02_guess / 3.0
    island_right = -2 * d02_guess / 3.0

    print(f"l=2 fit islands, anchored on the {K} fitted l=0 modes:")
    for i, nu_0_fitted in enumerate(best_nu0_arr):
        island = (x_bkg >= nu_0_fitted - island_left) & (x_bkg <= nu_0_fitted + island_right)
        mode_mask |= island
        print(f"  island {i}: anchor nu_0,{i} = {nu_0_fitted:.4f} µHz -> "
              f"[{nu_0_fitted - island_left:.4f}, {nu_0_fitted + island_right:.4f}] µHz, {int(island.sum())} bins")
    print(f"Total frequency bins used in the fit: {int(mode_mask.sum())}")

    x_fit = jnp.array(x_bkg[mode_mask])
    y_fit = jnp.array(y_bkg_obs[mode_mask])
    y_bkg_fixed_fit = jnp.array(y_bkg_fixed_full[mode_mask])

    bounds, bounds_dict = [], {'nu2': []}
    bounds.append((0, 90))
    bounds_dict['inc_2'] = (0, 90)
    bounds.append((0.0, 0.15))
    bounds_dict['nu_rot_2'] = (0.0, 0.15)
    bounds.append((0.1, 1.5))
    bounds_dict['V2'] = (0.1, 1.5)

    for nu_0_fitted in best_nu0_arr:
        nu2_g = nu_0_fitted - d02_guess
        low, high = nu2_g - d02_guess/4, nu2_g + d02_guess/4
        bounds.append((low, high))
        bounds_dict['nu2'].append((low, high))

    labels = [r"$i_{2}$ (deg)", r"$\Upsilon_{2}$", r"$V_2$"] + [rf"$\nu_{{2,{i}}}$" for i in range(K)]

    static_H0 = jnp.array(best_H0_arr)
    static_nu0 = jnp.array(best_nu0_arr)

    # --- Step 1: Differential Evolution ---
    print(f"\n--- Running DE (Popsize=100, Actual population={100*D}) ---")
    de_loss_batch = build_de_loss_batch(K, best_gamma0, static_H0, static_nu0, delta_f)
    t0_de = time.perf_counter()

    de_result = differential_evolution(
        neg_log_probability_vectorized, bounds,
        args=(de_loss_batch, x_fit, y_fit, y_bkg_fixed_fit, bounds, nyq),
        strategy='best1bin', maxiter=3000, popsize=100, tol=0.0001,
        mutation=(0.5, 1.0), recombination=0.7, disp=True,
        vectorized=True, updating='deferred'
    )
    best_fit_de = de_result.x
    print(f"DE Finished in {time.perf_counter() - t0_de:.1f} sec. NLL: {de_result.fun:.2f}")

    # --- Step 2: HMC (NUTS) ---
    dyn_warmup = 500 + (20 * D)
    dyn_samples = 1500 + (60 * D)
    print(f"\n--- Running NUTS (Warmup={dyn_warmup}, Samples={dyn_samples}) ---")
    t0_hmc = time.perf_counter()

    cos_i_2_init = jnp.clip(jnp.cos(best_fit_de[0] * jnp.pi / 180.0), 1e-6, 1.0 - 1e-6)
    init_values = {
        "cos_i_2": cos_i_2_init,
        "nu_rot_2": best_fit_de[1],
        "V2": best_fit_de[2],
        "nu2": jnp.array(best_fit_de[3:K+3])
    }
    init_strategy = numpyro.infer.init_to_value(values=init_values)
    kernel = NUTS(numpyro_model, target_accept_prob=0.90, init_strategy=init_strategy, dense_mass=True)

    model_kwargs = dict(x_fit=x_fit, y_bkg_fixed_fit=y_bkg_fixed_fit, nyq=nyq, K=K,
                        bounds_dict=bounds_dict, gamma0=best_gamma0, H0=static_H0, nu0=static_nu0,
                        delta_f=delta_f, y_obs=y_fit)

    hmc = MCMC(kernel, num_warmup=dyn_warmup, num_samples=dyn_samples, num_chains=1, progress_bar=False)
    nuts_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"NUTS seed: {nuts_seed}")
    rng_key = jax.random.PRNGKey(nuts_seed)
    hmc.run(rng_key, **model_kwargs, extra_fields=("diverging",))

    samples = hmc.get_samples()
    extra_fields = hmc.get_extra_fields()
    jax.block_until_ready((samples, extra_fields))

    print(f"HMC Finished in {time.perf_counter() - t0_hmc:.1f} sec.")

    # --- Sampling diagnostics ---
    t0_diag = time.perf_counter()
    print("\n--- Sampling Diagnostics (single chain; split R-hat) ---")
    divergences = extra_fields["diverging"].sum()

    flat_samples = np.column_stack([samples["inc_2"], samples["nu_rot_2"], samples["V2"], samples["nu2"]])

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
    r_hats = np.concatenate([np.atleast_1d(diag_summary[k]["r_hat"]) for k in ("inc_2", "nu_rot_2", "V2", "nu2")])
    n_effs = np.concatenate([np.atleast_1d(diag_summary[k]["n_eff"]) for k in ("inc_2", "nu_rot_2", "V2", "nu2")])

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

    median_vals = np.median(flat_samples, axis=0)

    # MAP: the posterior sample with the highest log-posterior
    t0_lp = time.perf_counter()
    log_post = compute_log_posterior_samples(samples, model_kwargs)
    n_bad_lp = int(np.sum(~np.isfinite(log_post)))
    if n_bad_lp == len(log_post):
        raise RuntimeError("Log-posterior is non-finite for every posterior sample.")
    best_idx = int(np.nanargmax(np.where(np.isfinite(log_post), log_post, np.nan)))
    map_hmc_params = flat_samples[best_idx]
    print(f"\nMAP sample: index {best_idx} of {len(log_post)} | "
          f"log-posterior = {log_post[best_idx]:.4f} "
          f"(evaluated for all samples in {time.perf_counter() - t0_lp:.2f} sec"
          + (f"; {n_bad_lp} non-finite values ignored" if n_bad_lp else "") + ")")

    print("\n--- Fixed L=0 Parameters ---")
    print(f"{'Gamma0 (uHz)':15s} : {best_gamma0:7.4f}")

    print("\n--- HMC Inferred Parameters ---")
    print(f"{'Parameter':<25} | {'Median':<9} | {'Uncertainty (+ / -)':<20} | {'MAP sample':<12} | {'In 1-Sigma'}")
    print("-" * 88)
    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        best_val = map_hmc_params[i]
        in_1sigma = q16 <= best_val <= q84
        print(f"{label:<25} | {q50:9.4f} | +{q84-q50:.4f} / -{q50-q16:.4f}     | {best_val:9.4f}   | {in_1sigma}")

    print(f"\n[TIMING] Diagnostics calculation took {time.perf_counter() - t0_diag:.2f} sec.")

    map_inc_2, map_nu_rot_2, map_V2 = map_hmc_params[0], map_hmc_params[1], map_hmc_params[2]
    map_nu2 = map_hmc_params[3:K+3]
    map_H2 = map_V2 * static_H0
    y_model_full_map = y_bkg_fixed_full + np.array(jax_modes(
        jnp.array(x_bkg), static_H0, static_nu0, jnp.array(map_H2), jnp.array(map_nu2),
        best_gamma0, map_inc_2, map_nu_rot_2, nyq, delta_f))

    med_inc_2, med_nu_rot_2, med_V2 = median_vals[0], median_vals[1], median_vals[2]
    med_nu2 = median_vals[3:K+3]
    med_H2 = med_V2 * static_H0
    y_model_full_med = y_bkg_fixed_full + np.array(jax_modes(
        jnp.array(x_bkg), static_H0, static_nu0, jnp.array(med_H2), jnp.array(med_nu2),
        best_gamma0, med_inc_2, med_nu_rot_2, nyq, delta_f))

    y_model_fit = np.array(y_bkg_fixed_fit + jax_modes(
        x_fit, static_H0, static_nu0, jnp.array(map_H2), jnp.array(map_nu2),
        best_gamma0, map_inc_2, map_nu_rot_2, nyq, delta_f))
    frac_res = np.array(y_fit / y_model_fit)
    mean_res, std_res = np.mean(frac_res), np.std(frac_res)

    print("\n--- Fractional Residuals (MAP Model) ---")
    print(f"Mean     = {mean_res:.4f} (Ideal ~ 1.0)")
    print(f"Std Dev = {std_res:.4f} (Ideal ~ 1.0)")

    # Plot ranges from the fitted modes
    mult_half = 2.0 * float(max(map_nu_rot_2, med_nu_rot_2)) + 3.0 * float(best_gamma0)
    l0_half = max(3.0 * float(best_gamma0), d02_guess / 3.0)
    island_lo = best_nu0_arr - island_left
    island_hi = best_nu0_arr + island_right
    l2_lo = np.minimum(map_nu2, med_nu2) - mult_half
    l2_hi = np.maximum(map_nu2, med_nu2) + mult_half
    l0_lo = best_nu0_arr - l0_half
    l0_hi = best_nu0_arr + l0_half
    mode_lo = np.minimum.reduce([island_lo, l2_lo, l0_lo])
    mode_hi = np.maximum.reduce([island_hi, l2_hi, l0_hi])
    wide_pad = 0.10 * dnu
    plot_lower = max(float(x_bkg[0]), float(mode_lo.min()) - wide_pad)
    plot_upper = min(float(x_bkg[-1]), float(mode_hi.max()) + wide_pad)

    print("\n--- Generating Output Document ---")
    with PdfPages(pdf_filename) as pdf:
        # 1. Corner plot
        fig_corner = corner.corner(
            flat_samples, labels=labels, truths=map_hmc_params, truth_color=CORNER_TRUTH_COLOR,
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
            nu_0_fitted = best_nu0_arr[i]
            z_pad = 0.25 * d02_guess
            z_lo = max(float(x_bkg[0]), float(mode_lo[i]) - z_pad)
            z_hi = min(float(x_bkg[-1]), float(mode_hi[i]) + z_pad)
            z_mask = (x_bkg >= z_lo) & (x_bkg <= z_hi)
            ax.axvspan(nu_0_fitted - island_left, nu_0_fitted + island_right, color=ISLAND_SHADE_COLOR,
                       alpha=ISLAND_SHADE_ALPHA, lw=0, zorder=Z_SHADE, label='Island window')
            ax.plot(x_bkg[z_mask], y_bkg_obs[z_mask], color=OBS_COLOR, alpha=0.6, lw=0.8, zorder=Z_OBS, label="Observations")
            ax.plot(x_bkg[z_mask], y_model_full_map[z_mask], color=MAP_FIT_COLOR, lw=MAP_FIT_LW + 0.2, zorder=Z_MAP, label="Total fit (HMC MAP)")
            ax.plot(x_bkg[z_mask], y_model_full_med[z_mask], color=MEDIAN_FIT_COLOR, linestyle=MEDIAN_FIT_LS,
                    lw=MEDIAN_FIT_LW, zorder=Z_MEDIAN, label="Total fit (HMC median)")
            ax.plot(x_bkg[z_mask], y_bkg_fixed_full[z_mask], color=BKG_COLOR, linestyle=BKG_LS, lw=1.6, zorder=Z_BKG, label="Fixed background")
            ax.axvline(nu_0_fitted, color=L0_LINE_COLOR, linestyle='-', lw=1.4, zorder=Z_VLINE, label=r"Fixed $\ell=0$")
            ax.axvline(map_nu2[i], color=L2_LINE_COLOR, linestyle=L2_LINE_LS, lw=2.0, zorder=Z_VLINE, label=r"MAP $\ell=2$ (center)")
            ax.set_xlim(z_lo, z_hi)
            z_top = max(np.max(y_bkg_obs[z_mask]), np.max(y_model_full_map[z_mask]), np.max(y_model_full_med[z_mask]))
            ax.set_ylim(0, 1.15 * z_top)
            ax.set_title(f"Island {i+1} near {nu_0_fitted:.2f} µHz", fontsize=ISLAND_TITLE_FONTSIZE)
            ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
            ax.legend(fontsize=ISLAND_LEGEND_FONTSIZE, ncol=3, loc='best')
        plt.tight_layout()
        pdf.savefig(fig_zoom)
        plt.close(fig_zoom)

        # 3. Wide-region comparison
        filt_mask = (x_bkg >= plot_lower) & (x_bkg <= plot_upper)
        fig_wide, ax_wide = plt.subplots(figsize=(11, 6))
        ax_wide.plot(x_bkg[filt_mask], y_bkg_obs[filt_mask], color=OBS_COLOR, alpha=0.45, lw=0.6, zorder=Z_OBS, label="Observations")
        ax_wide.plot(x_bkg[filt_mask], y_bkg_fixed_full[filt_mask], color=BKG_COLOR, linestyle=BKG_LS, lw=1.8, zorder=Z_BKG, label="Fixed background (MAP)")
        ax_wide.plot(x_bkg[filt_mask], y_model_full_map[filt_mask], color=MAP_FIT_COLOR, lw=MAP_FIT_LW, zorder=Z_MAP, label="Total fit (l=2 MAP + l=0 + background)")
        ax_wide.plot(x_bkg[filt_mask], y_model_full_med[filt_mask], color=MEDIAN_FIT_COLOR, linestyle=MEDIAN_FIT_LS,
                     lw=MEDIAN_FIT_LW, zorder=Z_MEDIAN, label="Total fit (l=2 median + l=0 + background)")
        for nu_0_fitted in best_nu0_arr:
            ax_wide.axvspan(nu_0_fitted - island_left, nu_0_fitted + island_right, color=ISLAND_SHADE_COLOR,
                            alpha=ISLAND_SHADE_ALPHA, lw=0, zorder=Z_SHADE)
        ax_wide.set_xlim(plot_lower, plot_upper)
        ax_wide.set_xlabel(r"Frequency ($\mu$Hz)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_wide.set_ylabel(r"PSD (ppm$^2/\mu$Hz)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_wide.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
        ax_wide.legend(loc='best', fontsize=PLOT_LEGEND_FONTSIZE, ncol=2)
        ax_wide.set_ylim(0, 1.2 * max(np.max(y_bkg_obs[filt_mask]), np.max(y_model_full_map[filt_mask]),
                                      np.max(y_model_full_med[filt_mask])))
        plt.tight_layout()
        pdf.savefig(fig_wide)
        plt.close(fig_wide)

        # 4. Fractional residuals in the islands
        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        ax_res.hist(frac_res, bins=50, density=True, color=RES_HIST_COLOR, alpha=0.75, label='Fractional Residuals')
        x_vals = np.linspace(0, np.max(frac_res), 200)
        ax_res.plot(x_vals, np.exp(-x_vals), color=RES_CURVE_COLOR, linestyle='--', lw=2, label=r'$e^{-x}$')
        ax_res.set_yscale('log')
        ax_res.set_ylim(bottom=1e-4, top=2)
        ax_res.set_xlabel("Fractional Residuals (Observed / Modelled)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_res.set_ylabel("Density", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax_res.set_title("Normalized Distribution of Fractional Residuals in Islands (MAP)", fontsize=PLOT_TITLE_FONTSIZE)
        ax_res.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
        ax_res.text(0.70, 0.95, f"Mean = {mean_res:.4f}\nStd Dev = {std_res:.4f}", transform=ax_res.transAxes,
                    fontsize=RESIDUAL_BOX_FONTSIZE, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_res.legend(loc='upper right', bbox_to_anchor=(0.98, 0.80), fontsize=PLOT_LEGEND_FONTSIZE)
        plt.tight_layout()
        pdf.savefig(fig_res)
        plt.close(fig_res)

    print(f"\nTotal Pipeline Time: {(time.perf_counter() - net_start)/60:.2f} minutes")
    print("==================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SPANDA-Stage 3b: quadrupole-mode (l=2) fit (DE + HMC) for one star.")
    ap.add_argument("kic", help="KIC number")
    ap.add_argument("stage1_csv", help="Stage 1 summary CSV used by the Stage 3a fit")
    ap.add_argument("stage3a_csv", help="Stage 3a summary CSV (stage3a_summary_<N>.csv)")
    ap.add_argument("--psd-dir", required=True, help="folder containing the power spectra")
    ap.add_argument("--psd-pattern", default="{kic}.npy", help="PSD file name with {kic} placeholder (default: {kic}.npy)")
    ap.add_argument("--outdir", default="./stage3b_fit_pdfs", help="folder for the PDF (default: ./stage3b_fit_pdfs)")
    args = ap.parse_args()
    main(args.kic, args.stage1_csv, args.stage3a_csv, args.psd_dir, args.psd_pattern, args.outdir)
