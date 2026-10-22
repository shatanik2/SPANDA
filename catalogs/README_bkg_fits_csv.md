# SPANDA-Stage 1 background fits: Kepler long-cadence red giants

File:

* `SPANDA_Stage1_Kepler_LC_bkg_fits_15792.csv`: background fits from SPANDA-Stage 1

This file is archived on Zenodo (doi:[10.5281/zenodo.23530405](https://doi.org/10.5281/zenodo.23530405);
access restricted until the publication of Bhattacharya et al., in prep.) and is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

This catalogue contains the background fits obtained with `SPANDA-Stage 1` for
**15,792** red-giant and red-clump stars observed by *Kepler* in long cadence.
There is one row per star, identified by its KIC number and sorted by it.

If you use this catalogue, please cite the SPANDA paper **Bhattacharya et al.** (reference to be added on publication).

## Model

Each power spectral density (PSD) was fitted with

$$ B(\nu) = W + \frac{a_1/b_1}{1+(\nu/b_1)^2} + \eta^2(\nu)\left[\frac{a_2/b_2}{1+(\nu/b_2)^4} + \frac{a_3/b_3}{1+(\nu/b_3)^4} + H_{\rm env}\exp\left(-\ln 2\,\frac{(\nu-\nu_{\rm max})^2}{{\rm HWHM}^2}\right)\right], $$

with $\eta^2(\nu) = {\rm sinc}^2[\pi\nu/(2\nu_{\rm Nyq})]$, using the likelihood for a
$\chi^2_2$-distributed PSD and uniform priors (ranges in Table 1 of the paper).
Initial values of $\nu_{\rm max}$ and $\Delta\nu$ were taken from Yu et al. (2018, ApJS, 236, 42).

The fit is a global Differential Evolution search followed by a single
No-U-Turn Sampler chain (NumPyro; 1000 warm-up and 3000 post-warm-up draws,
dense mass matrix, target acceptance 0.85), initialized at the Differential Evolution optimum.

## Columns

Missing or undefined values are written as `nan`.

### Identifier

| Column | Description |
|---|---|
| `kic` | *Kepler* Input Catalog identifier |

### Fitted parameters

Each of the 10 fitted parameters has four columns, `<prefix>_val`, `<prefix>_err_pos`,
`<prefix>_err_neg` and `<prefix>_map`:

| Suffix | Meaning |
|---|---|
| `_val` | posterior median (50th percentile) |
| `_err_pos` | 84th percentile minus median |
| `_err_neg` | median minus 16th percentile (given as a positive number) |
| `_map` | maximum a posteriori value: the posterior sample with the highest likelihood (equivalent to the highest posterior, as the priors are uniform) |

All four are in the units of the fitted quantity; for the logarithmic parameters,
the uncertainties are therefore in dex.

| Prefix | Fitted quantity | Underlying quantity | Unit of the underlying quantity |
|---|---|---|---|
| `nu_rmmax` | $\nu_{\rm max}$ |  $\nu_{\rm max}$  | µHz |
| `log_10_a_1_over_b_1` | $\log_{10}(a_1/b_1)$ | $a_1/b_1$ | ppm²/µHz |
| `log_10_b_1` | $\log_{10} b_1$ | $b_1$ | µHz |
| `log_10_a_2` | $\log_{10} a_2$ | $a_2$ | ppm² |
| `log_10_a_3` | $\log_{10} a_3$ | $a_3$ | ppm² |
| `log_10_b_2` | $\log_{10} b_2$ | $b_2$ | µHz |
| `log_10_b_3` | $\log_{10} b_3$ | $b_3$ | µHz |
| `log_10_W` | $\log_{10} W$ | $W$ | ppm²/µHz |
| `log_10_H_env` | $\log_{10} H_{\rm env}$ | $H_{\rm env}$ | ppm²/µHz |
| `HWHM` | half-width at half-maximum of the Gaussian envelope | HWHM | µHz |

Parameter values are given to 4 decimal places. The fitting range for these parameters can be found on [`README.MD IN STAGE1`](https://github.com/shatanik2/SPANDA/blob/main/Stage1/README.md)

### Sampling diagnostics

| Column | Description |
|---|---|
| `max_r_hat` | largest classical split- $\hat{R}$ over the 10 parameters, computed by comparing the two halves of the single chain (3 decimals) |
| `min_effective_samples` | smallest effective sample size $N_{\rm eff}$ over the 10 parameters (1 decimal) |
| `divergent_transitions` | number of divergent transitions among the 3000 post-warm-up draws |


### Residual statistics

The fractional residuals are $O(\nu_i)/B(\nu_i)$, the observed PSD divided by the
model above evaluated at the MAP parameters. Their mean and standard deviation
are computed over the frequency bins of three intervals.
The power-excess interval is $\nu^* _{\rm max} \pm 3.5\,{\rm HWHM}^*$, using the MAP values,
clipped to the observed frequency range. For an adequate background fit, the values marked with `_outside` are expected to be close to 1.

| Column | Frequency interval |
|---|---|
| `mean_res_full`, `std_res_full` | the entire PSD |
| `mean_res_outside`, `std_res_outside` | outside the power-excess interval |
| `mean_res_inside`, `std_res_inside` | inside the power-excess interval |

Residual statistics are given to 5 decimal places.

## Recommended quality selection

In the paper, a background fit is accepted for the subsequent p-mode analysis when

|1 − `mean_res_outside`| ≤ 0.02 and |1 − `std_res_outside`| ≤ 0.05,

which 15,337 of the 15,792 stars (97.12%) satisfy.
Stars failing this selection often show instrumental systematics or additional
signals (e.g. binarity, rotation, transits), and their fits should be used with caution.
