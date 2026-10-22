# SPANDA

**S**ampling-based **P**eak-b**A**gging for **N**ext generation **D**ata-driven **A**steroseismology

SPANDA is a hierarchical pipeline for automated background fitting and peak bagging of
solar-like oscillators. Each fitting stage combines a global Differential Evolution (DE)
search with Hamiltonian Monte Carlo sampling (No-U-Turn Sampler).

| Stage | Folder | What it does |
|---|---|---|
| 1 | [`Stage1/`](Stage1/) | Background fit: three Harvey-like components, white noise and a Gaussian power-excess envelope |
| 2 | [`Stage2/`](Stage2/) | Initialization: $`p`$-mode phase offset $`\epsilon_p`$ (2a) and radial-mode frequencies and heights (2b) |
| 3 | [`Stage3/`](Stage3/) | Peak bagging: radial ($`\ell = 0`$) modes (3a), then quadrupole ($`\ell = 2`$) modes (3b) |

Each stage takes the summary file of the previous stage as input. See the README in
each folder for its inputs, outputs and commands.

## Installation

Python 3.11. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for the reference package versions and for
installing JAX for your hardware (CPU, NVIDIA or AMD GPU, TPU). For the CPU route:

```bash
python -m pip install -r requirements_reproducibility.txt
```

All fits in the paper were run on a single CPU core; the fitting scripts set the thread
environment variables themselves.

## What you need to provide

1. **Power spectra**: one NumPy file per star, `<KIC>.npy` by default, containing frequency
   (in $`\mu\mathrm{Hz}`$) and power spectral density (in $`\mathrm{ppm}^2/\mu\mathrm{Hz}`$) as the
   first two columns of an array of shape $`(N, 2)`$ or larger (or the first two rows of a
   $`(2, N)`$ array) -- whichever dimension is longer is taken to be the frequency bins; any
   further columns are ignored. The frequencies must be on a uniform grid up to the Nyquist
   frequency. If your files are named differently, pass the name with `--psd-pattern`,
   e.g. `--psd-pattern "{kic}.npy.tmp.npy"`.
2. **Initial guesses**: a CSV file with columns `KIC`, `numax` and `Delnu` (in $`\mu\mathrm{Hz}`$),
   e.g. from a published catalog.

## Running the pipeline for one star

`run_single_star.sh` runs every step (Stage 1, its summary, Stages 2a and 2b, Stage 3a, Stage 3b
and their summaries) for one star. Provide the KIC number, the initial $`\nu_{\max}`$ and
$`\Delta\nu`$ (in $`\mu\mathrm{Hz}`$) and the folder containing the power spectrum:

```bash
chmod +x run_single_star.sh
./run_single_star.sh <KIC> <NUMAX> <DNU> <PSD_DIR> [PSD_PATTERN] [WORK_DIR]

# example, for KIC 8684542
./run_single_star.sh 8684542 178.0 13.43 /path/to/psds
# PSD files named differently, e.g. 8684542.npy.tmp.npy:
./run_single_star.sh 8684542 178.0 13.43 /path/to/psds "{kic}.npy.tmp.npy"
```

All results go to `runs/<KIC>/` (or `WORK_DIR`): the three summary CSVs
(`stage1_summary_1/`, `stage3a_summary_1/`, `stage3b_summary_1/`), the plots of every stage
and the full logs. The script stops with a message pointing to the relevant log if a step fails,
and reports whether the background fit passes the residual selection used in the paper.

## Running the full pipeline for many stars

Details for this part will be provided soon. Till then, you may try executing for each star using the scheme above.

## Notes

- The $`\ell = 2`$ fit (Stage 3b) checks that it is given the same Stage 1 summary as the
  $`\ell = 0`$ fit (Stage 3a).
- For short time series, the frequency resolution $`\delta f = 1/T`$ may exceed the mode
  linewidths; the linewidth is then not constrained by the data, and Stage 3 uncertainties are
  limited by the priors.
- **Pickled files.** Stage 2b saves its radial-mode estimates (`<KIC>_p_frequencies.npy`) as a
  Python dictionary, which NumPy stores using pickle; Stage 3a therefore reads these files with
  `np.load(..., allow_pickle=True)`. Loading a pickled file can execute arbitrary code, so only
  pass Stage 3a the `_p_frequencies.npy` files produced by your own Stage 2b runs, never files
  from an untrusted source. All other files (power spectra and Stage 2a estimates) are read
  with NumPy's default `allow_pickle=False`.

## Authors

Shatanik Bhattacharya, Sapnodip Pramanik, Jharnesh Verma, Meenakshi Gaira, Shravan Hanasoge

## Citation

If you use SPANDA, please cite the SPANDA paper (reference to be added on publication).
