# `dip/` — código propio de la tesis (pipeline Deep Image Prior)

No confundir con el fork upstream de *deep-image-prior*: las arquitecturas de red
y los helpers de imagen/tensor siguen en `models/` y `utils/` en la raíz y **no se
tocan**. Este paquete es sólo el código que escribimos nosotros encima.

Todo se corre **como módulo desde la raíz del repo**, para que `import models` /
`import utils` resuelvan sin trucos de `sys.path`:

```bash
./venv/bin/python -m dip.inpaint
./venv/bin/python -m dip.phase_diagram
./venv/bin/python -m dip.sample_points
```

## Módulos

| Módulo | Qué hace | Reemplaza a |
|---|---|---|
| `inpaint` | Runner de inpainting DIP (gris o RGB) sobre una imagen con máscara Bernoulli. Máscara → red `skip` → MSE sólo sobre pixeles observados → backtracking por caída de PSNR. Guarda snapshots, `metrics.csv`, figuras anotadas. | `restoration.py`, `restorationGRIS.py`, `restorationRGB.py` |
| `metrics` | PSNR / SSIM / MAE + mapa `|error|`. Wrappers finos sobre `skimage.metrics` y numpy. | (estaba inline en `restorationGRIS.py`) |
| `phase_diagram` | Genera el diagrama de fases SoC_max(log Ξ, log ℓ) con el modelo del continuo (`galpynostatic`). La imagen "original" densa que consume el barrido DIP. | `generate_phase_diagram.py` |
| `sample_points` | Elige un subconjunto disperso de puntos (ξ, ℓ) de la grilla del diagrama de fases para correr KMC real. | `sample_sparse_points.py` |

## `dip.inpaint` — variables de entorno

Todas opcionales, con defaults sanos en el script:

| Var | Default | Descripción |
|---|---|---|
| `IMAGE_PATH` | `./data/restoration/mapa_suave.png` | Imagen de entrada. |
| `OUTPUT_DIR` | `results/dip` | Carpeta de salida. |
| `N_CHANNELS` | `0` (inferir) | `1` = gris, `3` = RGB, `0` = según la imagen. |
| `MASK_FRAC` | `0.50` | Fracción **NO** observada (enmascarada). |
| `NUM_ITER` | `11000` | Iteraciones de optimización. |
| `LR` | `0.001` | Learning rate (Adam). |
| `REG_NOISE_STD` | `0.03` | Ruido de regularización sobre el input. |
| `SHOW_EVERY` | `100` | Cada cuántas iter se guarda snapshot + métricas. |
| `MAX_SIDE` | `0` | Si `>0`, redimensiona el lado mayor a ese valor antes de recortar. |
| `SEED` | — | Si se define, fija la semilla (máscara + init reproducibles). Sin definir = comportamiento histórico de `restorationGRIS.py`. |

### Salidas en `OUTPUT_DIR`

`mask.png` · `iter_XXXXX.png` (cada `SHOW_EVERY`) · `final_comparison.png`
(pixel-exacto) · `comparison_annotated.png` (con colorbar + PSNR/SSIM) ·
`psnr_curve.png` · `metrics.csv` · `restored.png` / `restored.npy` ·
`original.npy`.

## Barrido de máscaras (cluster)

`slurm/dip_sweep.slurm` corre `dip.inpaint` para
`MASK_FRAC ∈ {0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99}`:

```bash
sbatch slurm/dip_sweep.slurm                       # 128×128 (default)
RES=64 sbatch --export=ALL slurm/dip_sweep.slurm   # 64×64
```

Genera primero la imagen de entrada con `slurm/phase_diagram.slurm`
(o `./venv/bin/python -m dip.phase_diagram` en local).
