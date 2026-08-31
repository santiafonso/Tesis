# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This repo holds two unrelated bodies of work for the user's thesis (`Tesis`):

1. **Deep Image Prior (DIP) restoration experiments** — a fork of Dmitry Ulyanov's
   [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior) (CVPR 2018), extended with
   custom inpainting/restoration scripts at the repo root.
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

### Running restoration scripts

- `restoration.py`, `restorationRGB.py`, `restorationGRIS.py` are non-interactive `.py` exports of the
  `restoration.ipynb` notebook (script cells, no `%matplotlib inline`), meant to run headless on a SLURM
  cluster.
  - `restoration.py` — original grayscale ("barbara"/"mapa_suave") + "kate" pipeline, hardcoded paths and
    hyperparameters, outputs to `results_16G/`.
  - `restorationGRIS.py` — same as `restoration.py` but outputs to `results_gris/`.
  - `restorationRGB.py` — reworked color (3-channel) version, fully parameterized via environment
    variables instead of hardcoded constants, uses `matplotlib.use("Agg")` for headless plotting, and adds
    a fixed random seed. **This is the actively developed script** — prefer extending this one.
- `restorationRGB.py` environment variables (all optional, sane defaults in the script):
  `IMAGE_PATH`, `OUTPUT_DIR`, `MAX_SIDE` (0 = no resize), `MASK_FRAC`, `NUM_ITER`, `LR`, `REG_NOISE_STD`,
  `SHOW_EVERY`, `SEED`.
- Run directly: `./venv/bin/python restorationRGB.py`, or via the cluster job:
  `sbatch job.sh` (SLURM, requests 1 GPU, calls `restorationRGB.py`; edit the `cd` path at the top for
  the target user's home directory before submitting).
- Outputs (mask preview, periodic iteration snapshots, final restored image, `restored.npy`, PSNR) go to
  the script's `OUTPUT_DIR` (`results_16G/`, `results_gris/`, `results_rgb/`, etc. depending on script/env).

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
- `*.ipynb` at the root (`denoising.ipynb`, `inpainting.ipynb`, `super-resolution.ipynb`,
  `flash-no-flash.ipynb`, `feature_inversion.ipynb`, `activation_maximization.ipynb`, `sr_prior_effect.ipynb`,
  `restoration.ipynb`) are the original per-figure notebooks from the paper; `restoration.ipynb` is the one
  the root-level `restoration*.py` scripts were exported from and are meant to replace for headless runs.
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
- `run_kmc_param.slurm` — SLURM driver that sweeps a grid of `xi` × `el` × `NRUNS` combinations (edit the
  `xi_list`/`el_list`/`NRUNS` arrays at the top), and self-throttles to `$SLURM_NTASKS` concurrent
  `srun --exclusive` jobs using a named-pipe semaphore, stopping early if within `SAFETY_MARGIN` seconds of
  the partition's time limit. Submit with `sbatch run_kmc_param.slurm`.
- `parametros.dat` / `datos-*.dat` / `CS-40x40x40.xyz` are sample parameter logs and lattice
  configuration/output data from prior runs, not inputs required to build or run the code.

## Working across both parts

- Do not assume shared conventions between the two parts (e.g. Python formatting rules do not apply to the
  C++ code, and vice versa). When editing one, don't "clean up" or refactor the other unless asked.
- Large binary/reference files at the root (`*.pdf`, `*.xyz`, the `kmc_fast` binary) are reference papers
  and simulation artifacts, not something to regenerate or edit.
