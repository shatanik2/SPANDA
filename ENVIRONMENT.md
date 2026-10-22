# SPANDA — Environment and Installation Guide

This guide covers all three stages of SPANDA. It describes how to install the minimum
direct dependencies of SPANDA while selecting the JAX backend appropriate to your
CPU, GPU or other accelerator.

## Reference environment

The authors developed and tested SPANDA with the following packages and versions:

```
numpy==2.4.6
scipy==1.17.1
pandas==2.3.3
matplotlib==3.11.1
corner==2.3.0
jax==0.10.2
numpyro==0.21.0
```

JAX and NumPyro are used by the fitting scripts of Stage 1, Stage 3a and Stage 3b;
Stage 2 needs only NumPy, pandas and Matplotlib.

The authors used a system with Intel® Xeon® Gold 6526Y CPU. 
The execution referenced in the paper was performed with JAX forced to the CPU and each process restricted to one CPU core.

## Quick start

1. Create an environment with Python 3.11, for example:

   ```bash
   conda create -n spanda python=3.11
   conda activate spanda
   ```

2. Install the direct dependencies as described in
   [Hardware-appropriate installation](#hardware-appropriate-installation).

3. Reproduce the reference CPU execution, e.g. for Stage 1:

   Refer "Running the pipeline for one star" in [Main README](./README.md) 

---

## Hardware-appropriate installation

The minimal SPANDA dependency set is fixed, but JAX requires a hardware- and
platform-specific installation. Do **not** copy CUDA packages from an archived full
environment snapshot into a new environment unless exact reconstruction of the
reference environment is intended.

Installation is done in three steps, in this order:

- **Step 1:** the non-JAX, non-NumPyro direct dependencies;
- **Step 2:** JAX 0.10.2 with the backend appropriate to your hardware;
- **Step 3:** NumPyro 0.21.0.

NumPyro must be installed only **after** the desired JAX backend. NumPyro depends on JAX,
so installing it first would make pip install a default CPU-only JAX of whatever version
is newest, which may differ from 0.10.2.

### Step 1: non-JAX direct dependencies

```bash
python -m pip install \
    numpy==2.4.6 \
    scipy==1.17.1 \
    pandas==2.3.3 \
    matplotlib==3.11.1 \
    corner==2.3.0
```

### Step 2: JAX 0.10.2 (choose ONE of the following)

#### CPU

```bash
python -m pip install "jax==0.10.2"
```

No CUDA or ROCm packages are required.

#### NVIDIA GPU

Determine the available NVIDIA hardware and driver with `nvidia-smi`, and, for a locally
installed CUDA toolkit, the toolkit version with `nvcc --version`.

Use the JAX CUDA installation corresponding to the supported CUDA environment. Check the
JAX documentation before installation, because CUDA 12 and CUDA 13 have different driver
and GPU-generation requirements. JAX CUDA wheels are available for Linux only.

For the reference CUDA 12 route:

```bash
python -m pip install "jax[cuda12]==0.10.2"
```

For a CUDA 13 installation, use the corresponding extra only if JAX 0.10.2 provides the
required wheel for your Python version and platform:

```bash
python -m pip install "jax[cuda13]==0.10.2"
```

These commands download the NVIDIA CUDA runtime libraries as pip packages; only a
sufficiently recent NVIDIA driver is required on the system. If CUDA is already installed
system-wide (for example through an environment module), use the corresponding `-local`
extra so that the CUDA pip packages are not downloaded:

```bash
python -m pip install "jax[cuda12-local]==0.10.2"
# or
python -m pip install "jax[cuda13-local]==0.10.2"
```

If this exact JAX release is not available for your platform and Python version, do not
silently substitute a different JAX version when exact reproducibility is required. Use
the archived reference environment, or a compatible and explicitly documented JAX release.

JAX's CUDA extras install the accelerator-specific JAX components and the NVIDIA CUDA
runtime dependencies automatically. Do not list `jaxlib`, `jax-cuda12-plugin`,
`jax-cuda12-pjrt` or individual `nvidia-*` packages in a minimal dependency file.

#### AMD GPU

AMD GPU execution requires a compatible ROCm installation and the corresponding JAX ROCm
plugin. Install the ROCm software required by your system first (JAX does not install
ROCm itself); this route requires ROCm 7.x. Then:

```bash
python -m pip install "jax[rocm7-local]==0.10.2"
```

Each JAX ROCm plugin release is built against a specific ROCm version, which must match the
installed one; ROCm-specific fixes may be released as post-releases of the plugin. Consult
the current JAX/ROCm compatibility information, and if the command above does not resolve
for your ROCm version, follow AMD's JAX-on-ROCm instructions for the matching plugin.

#### Google Cloud TPU

On a Google Cloud TPU VM:

```bash
python -m pip install "jax[tpu]==0.10.2"
```

#### Apple GPU

JAX does not currently provide a native Mac GPU installation path comparable to NVIDIA or
AMD accelerators. Use the CPU installation unless a supported backend is documented for
your setup.

#### Windows

NumPyro has limited Windows support. Linux is preferred for reproducible SPANDA runs;
Windows users who need GPU execution should consider alternatives (like WSL2) and follow the current JAX and
NumPyro platform documentation.

### Step 3: NumPyro (after JAX, for every hardware choice)

```bash
python -m pip install "numpyro==0.21.0"
```

Do not add `--upgrade` (or `-U`), since that may allow pip to replace the JAX version
installed in Step 2.

### Verify the installation

```bash
python -m pip check
python -c "import jax, numpyro; print(jax.__version__, numpyro.__version__, jax.devices())"
```

The output should report jax 0.10.2 and numpyro 0.21.0 and list the expected device
(for example `CpuDevice`, `CudaDevice`, `RocmDevice` or `TpuDevice`).

### Running on an accelerator

The fitting scripts of Stage 1, Stage 3a and Stage 3b set `JAX_PLATFORMS="cpu"` near the
top of the file, so they run on the CPU regardless of the installed backend. To use an
accelerator, remove that line from the script, and omit `CUDA_VISIBLE_DEVICES=""` and
`JAX_PLATFORMS=cpu` from the run command.

---

## Reproducibility

`requirements_reproducibility.txt` pins the direct SPANDA dependencies (CPU route) and lets
pip resolve their transitive dependencies:

```bash
python -m pip install -r requirements_reproducibility.txt
```

This is the preferred minimal reproducibility specification. For other hardware, follow the
three steps above instead.

Neither a pinned dependency list nor a full environment snapshot guarantees bit-for-bit
numerical identity across operating systems, CPU or GPU architectures, drivers, CUDA/ROCm
installations or Python versions. In addition, the NUTS random seed is drawn afresh for
every run, so repeated runs should give statistically equivalent, not
identical, posterior samples.

## Scientific reference

The scientific and methodological details of SPANDA are described in the associated paper,
which should be cited for the method and in any publication that uses SPANDA:

Bhattacharya et al. (in prep.) — DOI and arXiv identifier to be provided.
