# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto de trabajo (mantener a mano)

- **Estado:** _(completar — qué capítulos/experimentos están listos y qué falta)_
- **Deadline:** _(completar)_
- **Estilo de citas / formato:** _(completar — APA, IEEE, etc.)_
- **Idioma de escritura:** español
- **NO hacer:** no reescribir secciones enteras sin pedirlo; proponer cambios primero.
- El libro (`~/ideas/libro-notas.md`) reusa material de acá — no mezclar los dos: la tesis es
  la forma académica, el libro es la versión narrativa.

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

- **`dip/restoration.py`** — the single DIP inpainting runner (**the actively developed one**; extend this).
  Merges the three old root scripts (`restoration.py`, `restorationGRIS.py`, `restorationRGB.py`).
  Method: Bernoulli mask → `skip` encoder-decoder → MSE on observed pixels only → PSNR-drop backtracking;
  crop to a multiple of 64, `ReflectionPad2d(1)`, `matplotlib.use("Agg")`. Grayscale or RGB via
  `N_CHANNELS` (`0` = infer from image). Env vars (all optional): `IMAGE_PATH`, `OUTPUT_DIR`,
  `N_CHANNELS`, `MASK_FRAC` (fraction *hidden*), `NUM_ITER`, `LR`, `REG_NOISE_STD` (↓ to ~`0.01` helps
  sharp phase fronts / negative `g`), `SHOW_EVERY` (metrics/CSV cadence), `SNAPSHOT_EVERY` (PNG-dump
  cadence, default = `SHOW_EVERY`), `PSNR_DROP_TOL` (dB drop in masked PSNR *vs its moving average*
  that triggers backtracking, default `-5.0`; **more negative = less aggressive** — use `-8`/`-10` on
  abrupt fronts), `MAX_FALLBACKS` (consecutive rollbacks before the state is accepted and the run
  continues, default `3`; stops a bad checkpoint from freezing the run), `MAX_SIDE` (0 = no resize),
  `SEED` (unset = historical non-seeded behaviour of `restorationGRIS.py`),
  `MASK_PATH` (optional `.npy` bool `H×W` fixed observed-pixel mask — overrides `MASK_FRAC`/Bernoulli;
  gets the same `ReflectionPad2d(1)` as the image).
  Outputs to `OUTPUT_DIR` (default `results/dip`): `mask.png`, `final_comparison.png`,
  `comparison_annotated.png`, `psnr_curve.png`, `snapshots_contact.png` (trajectory grid),
  `metrics.csv` (last rows: `final`, `mae`, `fallbacks`), `restored.png`/`.npy`, `original.npy`, and
  per-iteration `snapshots/iter_XXXXX.png` (in their own subdir, not the top level).
- **`dip/metrics.py`** — PSNR / SSIM / MAE + `|error|` map, lifted out of the runner. Thin wrappers over
  `skimage.metrics` and numpy. PSNR is the only metric that controls the algorithm (backtracking);
  SSIM/MAE are reporting only.
- **`dip/frontier_mask.py`** — builds non-uniform observed-pixel masks for the "points on the phase
  frontier vs not" experiment. Defines the frontier as smoothed, normalised `|∇SoC|`; samples `N`
  observed pixels per family — `frontier` (∝ `edge³`), `frontier_mix` (mostly frontier + `MIX_FRAC`
  uniform), `spread` (∝ `(1−edge)³`), `uniform` (Bernoulli). Nested across `OBS_FRACS`. Writes
  `data/restoration/masks/mf<hidden>/mask_<name>.npy` (+ `.png`, `preview.png`) — consumed by
  `dip.restoration` via `MASK_PATH`. Env: `IMAGE_PATH`, `SOC_NPY`, `OUT_DIR`, `OBS_FRACS`, `SEED`,
  `SMOOTH_SIGMA`, `EDGE_GAMMA`, `MIX_FRAC`. Fast, no GPU, runs local. Per-`g` masks (own `IMAGE_PATH` +
  `SOC_NPY` per map) are driven by `slurm/frontier_mask_g.slurm` → `data/restoration/masks_g/g<val>/`.
- **`dip/phase_diagram.py`** — generates the continuum-model phase diagram `SoC_max(log Ξ, log ℓ)` with
  `galpynostatic` (the dense "original" image the DIP sweep consumes). Env: `NUM_XI`, `NUM_ELL`,
  `GRID_SIZE`, `TIME_STEPS`, `VCUT` (φ_cut, def `-0.15`), `G` (Frumkin interaction param, def `0.0`;
  `g<0` attractive → sharper transition, `g>0` repulsive), `OUT_PNG`, `OUT_REF_PNG`, `OUT_NPY`.
  Fast-ish, no GPU, runs local (~2 min per 256 pts single-core, so 128² wants the cluster).
- **`slurm/phase_diagram_g.slurm`** — job array over `G` `{-4.0 … 0.0, 2.0, 4.0}` (one map per `g`,
  emphasis on negatives per the "los más conflictivos son los negativos" note); writes
  `results/phase_diagram_g/g<val>/{sim_<RES>.png,reference.png,soc.npy}`. Idempotent (skips if
  `soc.npy` exists). `RES=64` for a faster/coarser sweep.
- **`dip/sample_points.py`** — picks a sparse subset of `(ξ, ℓ)` grid points for real KMC runs. Env:
  `SEED`, `SAMPLE_FRAC`, `PHASE_DIR`.

### SLURM jobs (`slurm/`)

- `slurm/dip_restoration.slurm` — one DIP inpainting run (was `job.sh`).
- `slurm/dip_sweep.slurm` — mask-fraction sweep `{0.50 … 0.99}` (merges `run_gris_sweep.slurm` +
  `run_gris_sweep_128.slurm`); pick resolution with `RES=64|128` (default 128) → reads
  `data/restoration/phase_diagram_sim_${RES}.png`, writes `results/dip_sweep_${RES}/mf<frac>/`.
- `slurm/dip_frontier.slurm` — frontier-mask experiment: nested loop over `MFS`
  `{0.980 … 0.999}` × `MASKS` `{uniform, frontier, frontier_mix, spread}`, feeding
  `data/restoration/masks/mf<frac>/mask_<name>.npy` into `dip.restoration` via `MASK_PATH`; writes
  `results/dip_frontier/mf<frac>/<name>/`. Masks pre-generated with `dip.frontier_mask`.
- `slurm/frontier_mask_g.slurm` — builds per-`g` frontier masks (job array over `G`), one set per map
  from `results/phase_diagram_g/g<val>/{sim_<RES>.png,soc.npy}` → `data/restoration/masks_g/g<val>/mf<frac>/`.
  `OBS_FRACS` mirrors the `dip_gsweep_frontier` mask fractions. Needs the modern Python (`PYBIN`).
- `slurm/dip_gsweep_frontier.slurm` — the fix for the **negative-`g` reconstruction problem** (abrupt
  phase front → DIP blurs it, backtracking oscillates). Job array `9 g (negatives + 0.0) ×
  {frontier_mix, uniform} × 5 MASK_FRAC`, with `NUM_ITER=16000`, `REG_NOISE_STD=0.01`,
  `PSNR_DROP_TOL=-8` + `MAX_FALLBACKS=3` (robust backtracking); reads the per-`g` masks from
  `frontier_mask_g.slurm` (built with `MIX_FRAC=0.40` so plateaus keep anchors), writes
  `results/dip_gsweep_frontier/g<val>/mf<frac>/<name>/`. `uniform` is re-run at the same
  `N`/hyperparameters as an apples-to-apples baseline (do **not** compare against `results/dip_gsweep/`).
  Chain: `phase_diagram_g` → `frontier_mask_g` → `dip_gsweep_frontier` (via `--dependency=afterok:`).
  The first tanda (`MIX_FRAC=0.15`, `PSNR_DROP_TOL=-3`) is kept as `results/dip_gsweep_frontier_v1/`:
  `frontier_mix` only helped at `mf0.90 + g≤-2`, and thrashed (1000s of rollbacks) elsewhere.
- `slurm/phase_diagram.slurm` — runs `dip.phase_diagram` on the cluster (also fine locally without it).
- All `cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"` — no hardcoded home path to edit anymore.
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
  per-figure notebooks from the paper; `notebooks/restoration.ipynb` is the one `dip/restoration.py` was
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
dip/            our DIP thesis code (restoration, metrics, phase_diagram, sample_points) — run as -m dip.*
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
