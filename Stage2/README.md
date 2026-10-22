# Stage 2: initialization of the radial modes

Uses the Stage 1 background to find the $`p`$-mode phase offset $`\epsilon_p`$ (2a) and the
frequencies and heights of the radial modes (2b), which set the parameter ranges of Stage 3.
No sampling is done here.

Throughout, $`\nu^*_{\max}`$ and ${\rm HWHM}^*$ are the MAP values from Stage 1, $`\Delta\nu_{\rm init}`$
is the input large separation, and $`\mathcal{B}(\nu,\vec{\theta}_{\mathcal{B}}^{*})`$ is the MAP background of Stage 1 without the
Gaussian envelope ($`H_{\rm env} = 0`$).        

## Inputs

| Input | Option | Description |
|---|---|---|
| Stage 1 summary | `--stage1-csv` | `stage1_summary_<N>.csv` |
| Power spectra | `--psd-dir`, `--psd-pattern` | as in Stage 1 |

## Run, in this order

```bash
# 2a: epsilon_p for each star -> epsilon_p_estimates/<KIC>_epp_estimates.npy
python3 stage2a_epsilon_p.py <KIC> --stage1-csv <S1_CSV> --psd-dir /path/to/psds

# collect all 2a results into one table -> stage2a_epsilon_p_summary.txt
python3 stage2_collect_epsilon_p.py

# 2b: radial-mode frequencies and heights for each star -> p_mode_frequencies/<KIC>_p_frequencies.npy
python3 stage2b_radial_modes.py <KIC> --stage1-csv <S1_CSV> --psd-dir /path/to/psds
```

Run the collector after **all** 2a runs have finished, and again if any 2a run is repeated.

Options: `--plot` writes diagnostic PDFs (`stage2a_plots/`, `stage2b_plots/`); in 2b, `--alias`
also searches for Nyquist-aliased modes (off by default; still in development, use with caution),
and `--epsilon-summary` sets the table written by the collector.

## Outputs

| File | Content |
|---|---|
| `epsilon_p_estimates/<KIC>_epp_estimates.npy` | one row: `KEPLER_ID`, `OBS_MONTHS`, `nu_rmmax_map` ($`\nu^*_{\max}`$), `dnu_input` ($`\Delta\nu_{\rm init}`$), `HWHM_map` (${\rm HWHM}^*$), `EPSILON_P_CALC` ($`\epsilon_{p,\rm init}`$) |
| `stage2a_epsilon_p_summary.txt` | the same quantities for all stars, read by 2b |
| `p_mode_frequencies/<KIC>_p_frequencies.npy` | `radial_orders`, `nu_l0` ($`\nu^{\rm guess}_{0,i}`$, in $`\mu\mathrm{Hz}`$), `H_n0` ($`H^{\rm guess}_{0,i}`$, in $`\mathrm{ppm}^2/\mu\mathrm{Hz}`$), `epsilon_p` (the local phase of each order) and `is_aliased`, sorted by frequency |

The folder `p_mode_frequencies/` is the input of Stage 3a. Its files store a Python dictionary
and are read with `allow_pickle=True`; see the note on pickled files in the main README.
