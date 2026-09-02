# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This repo holds two unrelated bodies of work for the user's thesis (`Tesis`):

1. **Deep Image Prior (DIP) restoration experiments** — a fork of Dmitry Ulyanov's
   [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior) (CVPR 2018). The upstream
   network/util code stays in `models/` and `utils/`; our own thesis code lives in the `dip/` package
   and is run as `./venv/bin/python -m dip.<module>` from the repo root.
2. **KMC galvanostatic simulation** — a standalone, OpenMP-parallel C++ kinetic Monte Carlo simulation
   of electrochemical intercalation, run via SLURM on a cluster. Shares nothing with the Python code
   except the repo.

These live side by side with no shared code path; treat them as separate projects when working on either.

## Part 1 — Deep Image Prior / restoration

### Environment

- Python 3.6-targeted originally (`environment.yml` pins `pytorch=0.4`), but a `venv/` in the repo runs
  against modern PyTorch on Python 3.12 (see `__pycache__/*.cpython-312.pyc`). Use the existing `venv/`
  rather than the conda env file unless specifically reproducing the original paper's environment.
- Install via `pip install -r requirements.txt` (numpy, matplotlib, scipy, pillow, torch, torchvision, tqdm)
  or `conda env create -f environment.yml`.
- Docker: `nvidia-docker build -t deep-image-prior .` then
  `nvidia-docker run --rm -it --ipc=host -p 8888:8888 deep-image-prior` (starts a Jupyter server; note the
  Dockerfile re-clones the upstream repo rather than using local sources — treat it as a reference for
  dependency setup, not for running this repo's local scripts).

### `dip/` package — our thesis code

Run every module from the repo root so `import models` / `import utils` resolve:
`./venv/bin/python -m dip.<module>`. See `dip/README.md` for the full env-var tables.

- **`dip/inpaint.py`** — the single DIP inpainting runner (**the actively developed one**; extend this).
  Merges the three old root scripts (`restoration.py`, `restorationGRIS.py`, `restorationRGB.py`).
  Method: Bernoulli mask → `skip` encoder-decoder → MSE on observed pixels only → PSNR-drop backtracking;
  crop to a multiple of 64, `ReflectionPad2d(1)`, `matplotlib.use("Agg")`. Grayscale or RGB via
  `N_CHANNELS` (`0` = infer from image). Env vars (all optional): `IMAGE_PATH`, `OUTPUT_DIR`,
  `N_CHANNELS`, `MASK_FRAC` (fraction *hidden*), `NUM_ITER`, `LR`, `REG_NOISE_STD`, `SHOW_EVERY`,
  `MAX_SIDE` (0 = no resize), `SEED` (unset = historical non-seeded behaviour of `restorationGRIS.py`).
  Outputs to `OUTPUT_DIR` (default `results/dip`): `mask.png`, `iter_XXXXX.png`, `final_comparison.png`,
  `comparison_annotated.png`, `psnr_curve.png`, `metrics.csv`, `restored.png`/`.npy`, `original.npy`.
- **`dip/metrics.py`** — PSNR / SSIM / MAE + `|error|` map, lifted out of the runner. Thin wrappers over
  `skimage.metrics` and numpy. PSNR is the only metric that controls the algorithm (backtracking);
  SSIM/MAE are reporting only.
- **`dip/phase_diagram.py`** — generates the continuum-model phase diagram `SoC_max(log Ξ, log ℓ)` with
  `galpynostatic` (the dense "original" image the DIP sweep consumes). Env: `NUM_XI`, `NUM_ELL`,
  `GRID_SIZE`, `TIME_STEPS`, `OUT_PNG`, `OUT_REF_PNG`, `OUT_NPY`. Fast, no GPU, runs local.
- **`dip/sample_points.py`** — picks a sparse subset of `(ξ, ℓ)` grid points for real KMC runs. Env:
  `SEED`, `SAMPLE_FRAC`, `PHASE_DIR`.

### SLURM jobs (`slurm/`)

- `slurm/dip_inpaint.slurm` — one DIP inpainting run (was `job.sh`).
- `slurm/dip_sweep.slurm` — mask-fraction sweep `{0.50 … 0.99}` (merges `run_gris_sweep.slurm` +
  `run_gris_sweep_128.slurm`); pick resolution with `RES=64|128` (default 128) → reads
  `data/restoration/phase_diagram_sim_${RES}.png`, writes `results/dip_sweep_${RES}/mf<frac>/`.
- `slurm/phase_diagram.slurm` — runs `dip.phase_diagram` on the cluster (also fine locally without it).
- All three `cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"` — no hardcoded home path to edit anymore.
- `slurm/kmc/` — the KMC job scripts (Part 2), moved unchanged; still submitted from the repo root.

### Architecture (upstream deep-image-prior code)

- `models/` — network architectures used as the "prior": `skip.py` (the encoder-decoder with skip
  connections used by almost all experiments), `unet.py`, `resnet.py`, `texture_nets.py`, `dcgan.py`,
  `downsampler.py`. `models/__init__.py:get_net()` is the single factory used by all scripts/notebooks to
  build a network from a `NET_TYPE` string (`'skip'`, `'UNet'`, `'ResNet'`, `'texture_nets'`, `'identity'`).
- `utils/common_utils.py` — shared image/tensor conversion (`pil_to_np`, `np_to_pil`, `np_to_torch`,
  `torch_to_np`), `get_params`/`optimize` (the actual optimization loop driver), `crop_image`,
  `get_image_grid`/`plot_image_grid`.
- `utils/inpainting_utils.py`, `utils/denoising_utils.py`, `utils/sr_utils.py`,
  `utils/feature_inversion_utils.py`, `utils/perceptual_loss/` — task-specific helpers (mask generation,
  noise corruption, downsamplers, VGG-based perceptual loss matcher) used by the corresponding notebooks.
- The core DIP training loop pattern used everywhere: build a fixed random noise tensor as network input,
  optimize the network's weights (not the image) to reconstruct the corrupted/masked target image via MSE
  on the *observed* pixels only, track PSNR each `show_every`/`SHOW_EVERY` iterations, and optionally
  backtrack to a previous checkpoint if PSNR drops sharply (regularization against overfitting to noise).
- `notebooks/*.ipynb` (`denoising`, `inpainting`, `super-resolution`, `flash-no-flash`,
  `feature_inversion`, `activation_maximization`, `sr_prior_effect`, `restoration`) are the original
  per-figure notebooks from the paper; `notebooks/restoration.ipynb` is the one `dip/inpaint.py` was
  originally exported from and now replaces for headless runs.
- `data/` holds the sample images per task (`data/restoration/`, `data/inpainting/`, `data/denoising/`,
  `data/sr/`, `data/flash_no_flash/`, `data/feature_inversion/`).

## Part 2 — KMC galvanostatic simulation (C++)

- `KMC-Galvanostatic_noclus_param_claude.cpp` — single-file OpenMP KMC simulation of galvanostatic
  intercalation (no compile script/Makefile is checked in). Build manually, e.g.:
  ```
  g++ -O3 -fopenmp KMC-Galvanostatic_noclus_param_claude.cpp -o kmc_fast
  ```
- `acumulador_claude.h` — a Fenwick tree (`Acumulador<T>`) used by the simulation for O(log N)
  incremental rate-sum updates/searches instead of an O(N) rescan per KMC step; this is the main
  performance optimization in this version of the code (see the file header comment for the full list:
  incremental `VelocidadesAds`/`VelocidadesDif` updates, lazy `potencial()` recompute via
  `has_surface_neighbor[]`, no per-step cluster BFS).
- Run: `./kmc_fast <xi> <el> <numValue>` (galvanostatic parameter `xi`, lattice energy parameter `el`, and
  a run/trajectory index used to tag output files).
- `slurm/kmc/kmc_param.slurm` — SLURM driver that sweeps a grid of `xi` × `el` × `NRUNS` combinations
  (edit the `xi_list`/`el_list`/`NRUNS` arrays at the top), and self-throttles to `$SLURM_NTASKS`
  concurrent `srun --exclusive` jobs using a named-pipe semaphore, stopping early if within
  `SAFETY_MARGIN` seconds of the partition's time limit. Submit from the repo root:
  `sbatch slurm/kmc/kmc_param.slurm` (the `run_kmc_*.slurm` scripts moved into `slurm/kmc/` unchanged;
  they run in `$SLURM_SUBMIT_DIR` and expect the compiled binary at the repo root).
- `parametros.dat` / `datos-*.dat` / `CS-40x40x40.xyz` are sample parameter logs and lattice
  configuration/output data from prior runs, not inputs required to build or run the code.

## Repo layout

```
dip/            our DIP thesis code (inpaint, metrics, phase_diagram, sample_points) — run as -m dip.*
models/ utils/  upstream deep-image-prior fork — do not refactor
slurm/          job scripts: dip_*.slurm + phase_diagram.slurm; slurm/kmc/ for the C++ sim
notebooks/      upstream per-figure .ipynb
analysis/       one-off analysis scripts (job-specific, not part of the pipeline)
data/           input images (tracked)
papers/         reference PDFs (git-ignored)
archive/        old / superseded run-output dirs, kept locally, never committed (git-ignored)
presentaciones/ slide decks + their generators (the user's; leave alone)
KMC-*.cpp acumulador_claude.h   the KMC simulation (Part 2), plus its .dat/.xyz artifacts at root
```

New DIP runs write under `results/` (git-ignored); nothing there is precious — the images that back
the presentation live in `archive/dip_sweep_{64,128}/` and `archive/phase_diagram/`.

## Working across both parts

- Do not assume shared conventions between the two parts (e.g. Python formatting rules do not apply to the
  C++ code, and vice versa). When editing one, don't "clean up" or refactor the other unless asked.
- Large binary/artifact files at the repo root (`*.xyz`, `*.dat`, the `kmc_fast` binary) and the PDFs in
  `papers/` are reference papers and simulation artifacts — not something to regenerate or edit.
