# SPANDA-Stage 3 fits: radial and quadrupole p modes of Kepler red giants

Files:

* `SPANDA_Stage3a_Kepler_LC_l0_fits_15308.csv`: radial-mode ($`\ell = 0`$) fits from SPANDA-Stage 3a
* `SPANDA_Stage3b_Kepler_LC_l2_fits_15308.csv`: quadrupole-mode ($`\ell = 2`$) fits from SPANDA-Stage 3b

These files are archived on Zenodo (doi:[10.5281/zenodo.23530405](https://doi.org/10.5281/zenodo.23530405);
access restricted until the publication of Bhattacharya et al., in prep.) and are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Both files contain the same **15,308** stars, one row per star, identified by its KIC number
and sorted by it. These are the stars with a background fit (`SPANDA-Stage 1`) that passes
the residual selection |1 − `mean_res_outside`| ≤ 0.02 and |1 − `std_res_outside`| ≤ 0.05,
and with completed $\ell=0$ and $\ell=2$ fits. The background fits are provided in a
separate catalog.

If you use these catalogs, please cite the SPANDA paper (reference to be added on publication).

## Method in brief

The observed PSD is fitted with the fixed MAP background of `SPANDA-Stage 1` plus Lorentzian
mode profiles (or $\mathrm{sinc}^2$ profiles when the linewidth is below the resolution limit
$2\delta f/\pi$), using the likelihood for a $\chi^2_2$-distributed PSD evaluated only in narrow
windows around the modes, and uniform priors (Tables 2 and 3 of the paper). Each fit is a
Differential Evolution search followed by a single No-U-Turn Sampler chain.

- **Stage 3a** fits the common linewidth $\Gamma_0$ and the height $H_{0,i}$ and frequency
  $\nu_{0,i}$ of each radial mode $i$.
- **Stage 3b** keeps all Stage 3a parameters fixed at their MAP values and fits the
  $\ell=2$ modes, which share the linewidth $\Gamma_0$ and have heights $H_{2,i}=V_2H_{0,i}$.

## Value columns

Each fitted parameter has four columns:

| Suffix | Meaning |
|---|---|
| `_median` | posterior median (50th percentile) |
| `_err_pos` | 84th percentile minus median |
| `_err_neg` | median minus 16th percentile (given as a positive number) |
| `_map` | maximum a posteriori value: the posterior sample with the highest log-posterior |

Values are given to 4 decimal places, in the units of the fitted quantity (dex for
logarithmic parameters). Mode index $i$ runs from 0 upwards in order of increasing
frequency; an empty cell means that the star has no mode with that index.

## `SPANDA_Stage3a_Kepler_LC_l0_fits_15308.csv`

| Column | Description | Underlying quantity | Unit of underlying quantity |
|---|---|---|---|
| `kic` | *Kepler* Input Catalog identifier | -- | -- |
| `num_l0_modes` | number of fitted radial modes | -- | -- |
| `dnu_calculated` | local large separation used to set the fitting windows | `dnu_calculated` | µHz |
| `delta_f` | frequency resolution $\delta f$ of the PSD (see Caveats) | `delta_f` | µHz |
| `Gamma_0_*` | common $p$-mode linewidth $\Gamma_0$ | `Gamma_0_*` | µHz |
| `log_10_H_0_i_*` | $\log_{10}$ of the height $H_{0,i}$ of radial mode $i$ | `H_0_i_*` | ppm²/µHz |
| `nu_0_i_*` | frequency $\nu_{0,i}$ of radial mode $i$ | `nu_0_i_*` | µHz |
| `max_r_hat` | largest split- $\hat{R}$ over the fitted parameters | -- | -- |
| `min_n_eff` | smallest effective sample size over the fitted parameters | -- | -- |
| `divergent_transitions` | number of divergent transitions after warm-up | -- | -- |

## `SPANDA_Stage3b_Kepler_LC_l2_fits_15308.csv`

| Column | Description | Unit |
|---|---|---|
| `kic` | *Kepler* Input Catalog identifier | -- |
| `num_l2_modes` | number of fitted $\ell=2$ modes | -- |
| `d02_guess` | small separation used to set the $\ell=2$ windows | µHz |
| `delta_f` | frequency resolution $\delta f$ of the PSD (see Caveats) | µHz |
| `inc_2_deg_*` | inclination $i_{2\star}$ of the rotation axis | deg |
| `nu_rot_2_*` | rotational splitting $\Upsilon_2$ of the $\ell=2$ modes | µHz |
| `V_2_*` | visibility ratio $V_2=H_{2,i}/H_{0,i}$ | -- |
| `nu_2_i_*` | frequency $\nu_{2,i}$ of the $i$-th $\ell=2$ mode | µHz |
| `max_r_hat` | largest split- $\hat{R}$ over the fitted parameters | -- |
| `min_n_eff` | smallest effective sample size over the fitted parameters | -- |
| `divergent_transitions` | number of divergent transitions after warm-up | -- |

## Sampling diagnostics

`max_r_hat` is the largest classical split- $\hat{R}$ over the fitted parameters, computed by
comparing the two halves of the single chain; `min_n_eff` is the smallest effective sample size;
`divergent_transitions` is the number of divergent transitions after warm-up.

## Caveats

- **Frequency resolution.** `delta_f` is the frequency resolution $\delta f=1/T$ of the
  star's PSD, $T$ being the observation length. Below $2\delta f/\pi$ the mode profile
  no longer depends on $\Gamma_0$, which then cannot be measured, and for short time series
  the fitting windows contain only a few frequency bins. For such stars, $\Gamma_0$ and the
  frequency and height uncertainties are limited by the priors and should be used with care.
  Of the 15,308 stars, 100 have a typical linewidth
  $0.026+0.12\log_{10}(\Delta\nu/\mu\mathrm{Hz})$ $\mu\mathrm{Hz}$ below $2\delta f/\pi$.
- **Inclination and splitting.** The rotationally split $\ell=2$ components overlap when
  $\Upsilon_2\lesssim\max(\Gamma_0,\delta f)$; the inclination and splitting are then poorly
  constrained and strongly correlated.
- The uncertainties of the Stage 3b parameters are conditional on the fixed Stage 3a MAP
  values and on the fixed Stage 1 background.
