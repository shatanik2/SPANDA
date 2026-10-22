# Stage 3: radial and quadrupole $`p`$ modes

Island fits of the $`p`$ modes on the fixed Stage 1 background, each with Differential Evolution
followed by a single NUTS chain. The mode profiles (Lorentzian, or $`{\rm sinc}^2`$ when the linewidth
is below the resolution limit $`2\,\delta f/\pi`$), the likelihood, the fitting windows and the
prior choices are described in Bhattacharya et al. (in prep.).

- **3a** fits the common linewidth $`\Gamma_0`$ and the height and frequency of each radial
  ($`\ell = 0`$) mode.
- **3b** keeps all 3a parameters fixed (at their MAP values) and fits the $`\ell = 2`$ frequencies,
  the visibility $`V_2`$, the inclination $`i_{2\star}`$ and the rotational splitting $`\Upsilon_2`$.

## 3a: radial modes

```bash
mkdir -p logs_l0
python3 stage3a_l0_fit.py <KIC> <S1_CSV> ../Stage2/p_mode_frequencies --psd-dir /path/to/psds > logs_l0/kic_<KIC>.log 2>&1
python3 make_stage3a_summary.py --logdir logs_l0
```

Inputs: the Stage 1 summary and the Stage 2 folder `p_mode_frequencies/`.

Uniform priors, for the radial modes $`i`$ with Stage 2 frequencies within
$`\nu^*_{\max} \pm 2\max({\rm HWHM}^*, \Delta\nu_{\rm init})`$. Here $`\nu^{\rm guess}_{0,i}`$ and
$`H^{\rm guess}_{0,i}`$ are the Stage 2 estimates, and
$`\Gamma^{(3a)}_{\rm guess} = 0.026 + 0.12\log_{10}(\langle\Delta\nu\rangle_{(3a)}/\mu{\rm Hz})\ \mu{\rm Hz}`$,
with $`\langle\Delta\nu\rangle_{(3a)}`$ the mean spacing of the four Stage 2 frequencies closest to $`\nu^*_{\max}`$.

| Parameter | Unit | Fitted parameter | Lower bound | Upper bound | Summary column prefix |
|---|---|---|---|---|---|
| $`\Gamma_0`$ | $`\mu\mathrm{Hz}`$ | $`\Gamma_0`$ | $`\min(0.02,\ \Gamma^{(3a)}_{\rm guess}/4)`$ | $`\min(0.25,\ 4\,\Gamma^{(3a)}_{\rm guess})`$ | `Gamma_0` |
| $`H_{0,i}`$ | $`\mathrm{ppm}^2/\mu\mathrm{Hz}`$ | $`\log_{10} H_{0,i}`$ | $`\log_{10}(0.01\,H^{\rm guess}_{0,i})`$ | $`\log_{10}(100\,H^{\rm guess}_{0,i})`$ | `log_10_H_0_i` |
| $`\nu_{0,i}`$ | $`\mu\mathrm{Hz}`$ | $`\nu_{0,i}`$ | $`\nu^{\rm guess}_{0,i} - \Gamma^{(3a)}_{\rm guess}`$ | $`\nu^{\rm guess}_{0,i} + \Gamma^{(3a)}_{\rm guess}`$ | `nu_0_i` |

Output: `stage3a_fit_pdfs/l0_fit_DE_HMC_<KIC>.pdf` per star, and `stage3a_summary_<M>/` with
`stage3a_summary_<M>.csv` (median, errors and MAP of each parameter, diagnostics, and the
frequency resolution $`\delta f`$ as `delta_f`), a diagnostics table (`.tex`), run-time histograms
and diagnostics binned in $`\langle\Delta\nu\rangle_{(3a)}`$.

## 3b: quadrupole modes

```bash
mkdir -p logs_l2
python3 stage3b_l2_fit.py <KIC> <S1_CSV> stage3a_summary_<M>/stage3a_summary_<M>.csv --psd-dir /path/to/psds > logs_l2/kic_<KIC>.log 2>&1
python3 make_stage3b_summary.py --logdir logs_l2
```

Use the **same** Stage 1 summary as in 3a; the script checks this and stops otherwise.

Uniform priors, with $`\nu^{\rm fitted}_{0,i}`$ the MAP radial-mode frequencies from 3a and
$`d_{02} = 0.121\,\langle\Delta\nu\rangle_{(3a)} + 0.047\ \mu{\rm Hz}`$. The $`\ell = 2`$ modes share
the linewidth $`\Gamma_0`$ and have heights $`H_{2,i} = V_2\,H_{0,i}`$.

| Parameter | Unit | Fitted parameter | Lower bound | Upper bound | Summary column prefix |
|---|---|---|---|---|---|
| $`i_{2\star}`$ | deg | $`\cos i_{2\star}`$ | $`0`$ | $`1`$ | `inc_2` (reported in deg) |
| $`\Upsilon_2`$ | $`\mu\mathrm{Hz}`$ | $`\Upsilon_2`$ | $`0`$ | $`0.15`$ | `nu_rot_2` |
| $`V_2`$ | -- | $`V_2`$ | $`0.1`$ | $`1.5`$ | `V2` |
| $`\nu_{2,i}`$ | $`\mu\mathrm{Hz}`$ | $`\nu_{2,i}`$ | $`\nu^{\rm fitted}_{0,i} - 5\,d_{02}/4`$ | $`\nu^{\rm fitted}_{0,i} - 3\,d_{02}/4`$ | `nu_2_i` |

Output: `stage3b_fit_pdfs/l2_fit_DE_HMC_<KIC>.pdf` per star, and `stage3b_summary_<L>/` with
`stage3b_summary_<L>.csv` and the same diagnostic products as in 3a.

## Summary columns

Each fitted parameter has the columns `_median`, `_err_pos` (84th percentile minus median),
`_err_neg` (median minus 16th percentile), `_map` (highest-log-posterior sample) and `_in_1sigma`,
in the units of the fitted parameter (dex for the logarithmic heights). Mode index $`i`$ increases
with frequency. Diagnostics: `max_r_hat` (largest split-$`\hat{R}`$ of the single chain),
`min_n_eff` (smallest effective sample size) and `divergent_transitions`.

## Caveats

- When the modes are unresolved ($`\Gamma_0 < 2\,\delta f/\pi`$, e.g. for short time series),
  $`\Gamma_0`$ is not constrained by the data and the uncertainties of the other parameters are
  limited by the priors.
- When the rotationally split components overlap ($`\Upsilon_2 \lesssim \max(\Gamma_0, \delta f)`$),
  $`i_{2\star}`$ and $`\Upsilon_2`$ are poorly constrained and should be used with caution.
- The 3b uncertainties are conditional on the fixed 3a MAP values and on the fixed Stage 1 background.

See Bhattacharya et al. (in prep.) for details.
