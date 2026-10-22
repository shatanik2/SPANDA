# Stage 1: background fit

Fits the power spectrum of each star with three Harvey-like profiles, white noise and a Gaussian
power-excess envelope (10 parameters, uniform priors set from the initial $`\nu_{\max}`$ and
$`\Delta\nu`$), using Differential Evolution followed by a single NUTS chain (1000 warm-up,
3000 samples).

## Model

The observed power spectral density $`O(\nu)`$ is fitted with

```math
\mathcal{B}(\nu) = W + \frac{a_1/b_1}{1+(\nu/b_1)^2}
+ \eta^2(\nu)\left[\frac{a_2/b_2}{1+(\nu/b_2)^4} + \frac{a_3/b_3}{1+(\nu/b_3)^4}
+ H_{\rm env}\exp\left(-\ln 2\,\frac{(\nu-\nu_{\max})^2}{{\rm HWHM}^2}\right)\right],
```

where the attenuation factor is

```math
\eta^2(\nu) = {\rm sinc}^2\left(\frac{\pi\nu}{2\,\nu_{\rm Nyq}}\right),
```

with $`\nu_{\rm Nyq}`$ the highest frequency of the power spectrum. The log-likelihood, for a
$`\chi^2`$ distribution with two degrees of freedom, is evaluated over all frequency bins $`\nu_i`$:

```math
\ln\mathcal{L} = -\sum_i \left[\ln \mathcal{B}(\nu_i) + \frac{O(\nu_i)}{\mathcal{B}(\nu_i)}\right].
```

## Fitted parameters and priors

All priors are uniform in the fitted parameter. Here $`\nu_{\max,\rm init}`$ and
$`\Delta\nu_{\rm init}`$ are the initial values from the input catalog (in $`\mu\mathrm{Hz}`$), and
$`H_{\max}`$ is the maximum of the power spectrum over $`\nu_{\max,\rm init} \pm 3.5\,\Delta\nu_{\rm init}`$.

| Underlying parameter | Unit | Fitted parameter | Lower bound | Upper bound | Summary column prefix |
|---|---|---|---|---|---|
| $`\nu_{\max}`$ | $`\mu\mathrm{Hz}`$ | $`\nu_{\max}`$ | $`\nu_{\max,\rm init} - 2\,\Delta\nu_{\rm init}`$ | $`\nu_{\max,\rm init} + 2\,\Delta\nu_{\rm init}`$ | `nu_rmmax` |
| $`a_1/b_1`$ | $`\mathrm{ppm}^2/\mu\mathrm{Hz}`$ | $`\log_{10}(a_1/b_1)`$ | $`-1`$ | $`10`$ | `log_10_a_1_over_b_1` |
| $`b_1`$ | $`\mu\mathrm{Hz}`$ | $`\log_{10} b_1`$ | $`-1`$ | $`1`$ | `log_10_b_1` |
| $`a_2`$ | $`\mathrm{ppm}^2`$ | $`\log_{10} a_2`$ | $`\log_{10}\left[\frac{2\sqrt{2}}{\pi}\left(3382\,\nu_{\max,\rm init}^{-0.609} \times 0.5\right)^2\right]`$ | $`\log_{10}\left[\frac{2\sqrt{2}}{\pi}\left(3382\,\nu_{\max,\rm init}^{-0.609} \times 1.5\right)^2\right]`$ | `log_10_a_2` |
| $`a_3`$ | $`\mathrm{ppm}^2`$ | $`\log_{10} a_3`$ | $`\log_{10}\left[\frac{2\sqrt{2}}{\pi}\left(3382\,\nu_{\max,\rm init}^{-0.609} \times 0.5\right)^2\right]`$ | $`\log_{10}\left[\frac{2\sqrt{2}}{\pi}\left(3382\,\nu_{\max,\rm init}^{-0.609} \times 1.5\right)^2\right]`$ | `log_10_a_3` |
| $`b_2`$ | $`\mu\mathrm{Hz}`$ | $`\log_{10} b_2`$ | $`\log_{10}\left(0.317\,\nu_{\max,\rm init}^{0.970} \times 0.6\right)`$ | $`\log_{10}\left(0.317\,\nu_{\max,\rm init}^{0.970} \times 1.4\right)`$ | `log_10_b_2` |
| $`b_3`$ | $`\mu\mathrm{Hz}`$ | $`\log_{10} b_3`$ | $`\log_{10}\left(0.948\,\nu_{\max,\rm init}^{0.992} \times 0.6\right)`$ | $`\log_{10}\left(0.948\,\nu_{\max,\rm init}^{0.992} \times 2.0\right)`$ | `log_10_b_3` |
| $`W`$ | $`\mathrm{ppm}^2/\mu\mathrm{Hz}`$ | $`\log_{10} W`$ | $`-0.5`$ | $`5`$ | `log_10_W` |
| $`H_{\rm env}`$ | $`\mathrm{ppm}^2/\mu\mathrm{Hz}`$ | $`\log_{10} H_{\rm env}`$ | $`\log_{10}(0.01\,H_{\max})`$ | $`\log_{10}(0.5\,H_{\max})`$ | `log_10_H_env` |
| HWHM | $`\mu\mathrm{Hz}`$ | HWHM | $`0.5\,\Delta\nu_{\rm init}`$ | $`4.0\,\Delta\nu_{\rm init}`$ | `HWHM` |

In the summary CSV, each prefix is followed by `_val` (posterior median), `_err_pos`
(84th percentile minus median), `_err_neg` (median minus 16th percentile), `_map` (posterior
sample with the highest likelihood) and `_in_1sigma`. All are in the units of the fitted
parameter; for the logarithmic parameters the uncertainties are therefore in dex.

## Inputs

| Input | Option | Description |
|---|---|---|
| KIC number | positional | star to fit |
| Initial guesses | `--catalog` | CSV with columns `KIC`, `numax`, `Delnu` (in $`\mu\mathrm{Hz}`$) |
| Power spectra | `--psd-dir`, `--psd-pattern` | folder and file name (default `{kic}.npy`) |

## 1. Fit each star

```bash
mkdir -p logs
python3 stage1_background_fit.py <KIC> --catalog init.csv --psd-dir /path/to/psds > logs/kic_<KIC>.log 2>&1
```

Output: `stage1_fit_pdfs/bkg_fit_DE_HMC_<KIC>.pdf` (corner plot, fits, residuals, traces; change
with `--outdir`) and the log with all fitted values and diagnostics.

## 2. Build the summary

```bash
python3 make_stage1_summary.py --logdir logs --catalog init.csv --psd-dir /path/to/psds
```

Pass `--catalog` so that the input $`\nu_{\max}`$ and $`\Delta\nu`$ keep full precision (the logs
print 2 decimals). The Nyquist frequency for the plots is read from the PSDs with `--psd-dir`;
alternatively give `--nyquist <value>` (in $`\mu\mathrm{Hz}`$) or `--mission kepler --cadence LC`.

Output, in `stage1_summary_<N>/` ($`N`$ = number of completed stars):

| File | Content |
|---|---|
| `stage1_summary_<N>.csv` | per star: inputs, run times, diagnostics, median, errors and MAP of each parameter, residual statistics |
| `stage1_diagnostics_table_<N>.tex` | diagnostic and residual tiers as an AASTeX table |
| `Stage1_DE_HMC_times_<N>.pdf` | run-time histograms |
| `Stage1_diagnostics_vs_numax_<N>.pdf` | diagnostics binned in $`\nu_{\max}`$ and in $`\Delta\nu/(0.263\,\nu_{\max}^{0.772})`$ |
| `plot_details_<N>.txt` | details of the plots |

`stage1_summary_<N>.csv` is the input of Stages 2 and 3.
