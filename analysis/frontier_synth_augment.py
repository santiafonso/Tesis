#!/usr/bin/env python
# coding: utf-8
"""Idea 15/9 (fuera de los 5 puntos): en vez de gastar presupuesto de puntos
REALES en la meseta (donde ya sabemos mas o menos que valor va, se acerca a
una constante), se "inventan" puntos SINTETICOS ahi con un valor inferido
-- y todo el presupuesto real se concentra en la frontera, la unica zona
donde de verdad no se sabe que pasa.

Mecanismo para inventar el valor (reusa la logica de la rama parqueada
`curvefit-frontier`, sin red neuronal, solo con los puntos reales):
  1. Clasificar los puntos reales en meseta alta/baja (casi bimodal).
  2. Estimar la curva de frontera emparejando cada punto con su vecino mas
     cercano de la clase opuesta (funciona con pocos puntos).
  3. Ajustar una relajacion exponencial C + A*exp(-d/L) por lado (cada
     meseta NO es plana, satura despacio lejos de la frontera).
  4. Para pixeles lejos de la curva estimada (mas de MARGIN px), asignarles
     el valor que predice esa relajacion -- NO el valor real (no se mira la
     imagen ahi, solo se infiere de los puntos reales).

Salida: una mascara AUMENTADA (reales + sinteticos) y una imagen COMPUESTA
(valores reales en los puntos reales, valores inventados en los sinteticos)
listas para alimentar dip.restoration via IMAGE_PATH/MASK_PATH. Los puntos
sinteticos son "gratis" (no cuentan como presupuesto real de KMC).

Uso:
    ./venv/bin/python -m analysis.frontier_synth_augment -4.0 0.980 frontier

Escribe en data/restoration/masks_g/g<g>/mf<mf>/synth_<family>/:
    image_composite.png   mask_augmented.npy   preview.png
e imprime cuantos puntos reales/sinteticos quedaron.

Variables de entorno (opcionales):
    N_SYNTH    cuantos puntos sinteticos como maximo (def: 400)
    MARGIN     distancia minima (px) a la curva estimada para poner un
               punto sintetico (def: 6.0) -- evita inventar cerca del borde,
               donde el modelo de relajacion es menos confiable.
"""
from __future__ import print_function

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from scipy.optimize import curve_fit

N_SYNTH = int(os.environ.get("N_SYNTH", "400"))
MARGIN = float(os.environ.get("MARGIN", "6.0"))
SEED = int(os.environ.get("SEED", "42"))

PHASE_DIAGRAM_DIR = "results/phase_diagram_g"
MASKS_DIR = "data/restoration/masks_g"


def load_ground_truth(g):
    """Mismo volteo que dip.phase_diagram usa para el PNG -- para que las
    filas coincidan con la mascara y con lo que ve dip.restoration."""
    path = os.path.join(PHASE_DIAGRAM_DIR, "g%s" % g, "soc.npy")
    return np.flipud(np.load(path)).astype(np.float32).copy()


def find_level_threshold(vals):
    sv = np.sort(vals)
    if len(sv) < 2:
        return float(np.median(vals))
    gaps = np.diff(sv)
    k = int(np.argmax(gaps))
    return float((sv[k] + sv[k + 1]) / 2.0)


def classify_real_points(mask, gt):
    """Clasifica los puntos REALES en meseta alta/baja (casi bimodal) --
    sigue haciendo falta para saber, punto por punto, a que lado de la
    relajacion exponencial pertenece cada uno."""
    rows, cols = np.where(mask)
    vals = gt[rows, cols]
    thresh = find_level_threshold(vals)
    is_high = vals >= thresh
    high_rc = np.column_stack([rows[is_high], cols[is_high]]).astype(float)
    low_rc = np.column_stack([rows[~is_high], cols[~is_high]]).astype(float)
    if len(high_rc) == 0 or len(low_rc) == 0:
        raise ValueError("la mascara solo observa una meseta -- no se puede ubicar la frontera")
    return thresh, high_rc, low_rc


def gradient_ridge_curve(gt, smooth_sigma=1.0, min_peak=0.05):
    """Curva de frontera = cresta de |grad soc| fila por fila (argmax con
    refinamiento sub-pixel), MISMO metodo ya arreglado hoy en
    analysis/family_montage.py y dip/frontier_mask.py (el que antes
    contorneaba por debajo del pico y dibujaba una falsa doble linea/Z).
    Usa el soc.npy denso completo, no los puntos observados -- consistente
    con como dip/frontier_mask.py ya arma las familias frontier/frontier_mix/
    spread (tambien mira el soc.npy completo para decidir donde muestrear)."""
    gy, gx = np.gradient(ndi.gaussian_filter(gt, smooth_sigma))
    gmag = np.hypot(gx, gy)
    gmag = ndi.gaussian_filter(gmag, smooth_sigma)
    edge = gmag / max(gmag.max(), 1e-12)

    H, W = edge.shape
    xs = np.full(H, np.nan)
    for y in range(H):
        row = edge[y]
        k = int(np.argmax(row))
        if row[k] < min_peak:
            continue
        if 0 < k < W - 1:
            v0, v1, v2 = row[k - 1], row[k], row[k + 1]
            denom = v0 - 2 * v1 + v2
            delta = float(np.clip(0.5 * (v0 - v2) / denom, -1, 1)) if denom != 0 else 0.0
        else:
            delta = 0.0
        xs[y] = k + delta
    # filas sin cresta clara (mesetas planas de punta a punta): mantener el
    # valor mas cercano en vez de NaN, para no dejar agujeros en la curva.
    good = ~np.isnan(xs)
    if good.any():
        idx = np.arange(H)
        xs = np.interp(idx, idx[good], xs[good])
    return xs


def _exp_safe(x):
    return np.exp(np.clip(x, -50, 50))


def _relax_func(d, C, A, L):
    return C + A * _exp_safe(-d / L)


def fit_relaxation(dist, vals):
    dist = np.asarray(dist, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 4 or np.ptp(dist) < 1e-6:
        return float(np.median(vals)), 0.0, 1.0
    far = dist >= np.percentile(dist, 75)
    near = dist <= np.percentile(dist, 25)
    C0 = float(np.median(vals[far])) if far.any() else float(np.median(vals))
    A0 = float(np.median(vals[near]) - C0) if near.any() else 0.0
    L0 = max(float(np.median(dist)), 1.0)
    try:
        (C, A, L), _ = curve_fit(
            _relax_func, dist, vals, p0=[C0, A0, L0],
            bounds=([0.0, -1.0, 1e-3], [1.0, 1.0, 500.0]), maxfev=5000,
        )
        return float(C), float(A), float(L)
    except RuntimeError:
        return C0, A0, L0


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    g, mf, family = sys.argv[1], sys.argv[2], sys.argv[3]

    gt = load_ground_truth(g)
    H, W = gt.shape
    mask_path = os.path.join(MASKS_DIR, "g%s" % g, "mf%s" % mf, "mask_%s.npy" % family)
    mask = np.load(mask_path)
    n_real = int(mask.sum())

    thresh, high_rc, low_rc = classify_real_points(mask, gt)
    xs = np.clip(gradient_ridge_curve(gt), 0, W - 1)
    high_left = bool(high_rc[:, 1].mean() <= low_rc[:, 1].mean())

    xs_at_high = xs[high_rc[:, 0].astype(int)]
    xs_at_low = xs[low_rc[:, 0].astype(int)]
    dist_high_obs = (xs_at_high - high_rc[:, 1]) if high_left else (high_rc[:, 1] - xs_at_high)
    dist_low_obs = (low_rc[:, 1] - xs_at_low) if high_left else (xs_at_low - low_rc[:, 1])
    high_fit = fit_relaxation(dist_high_obs, gt[high_rc[:, 0].astype(int), high_rc[:, 1].astype(int)])
    low_fit = fit_relaxation(dist_low_obs, gt[low_rc[:, 0].astype(int), low_rc[:, 1].astype(int)])
    print("Relajacion alta  C=%.4f A=%.4f L=%.2f" % high_fit)
    print("Relajacion baja  C=%.4f A=%.4f L=%.2f" % low_fit)

    # --- candidatos sinteticos: GRILLA regular (no azar) lejos de la curva
    # estimada, no ya reales -- misma idea que la familia `grid` (reparto
    # parejo, sin huecos ni amontonamientos), aplicada a los inventados.
    rng = np.random.default_rng(SEED)
    step = max(1, int(round((H * W / max(N_SYNTH, 1)) ** 0.5)))
    off = step // 2
    grid = np.zeros((H, W), dtype=bool)
    grid[off::step, off::step] = True
    dist_to_curve = np.abs(np.arange(W)[None, :] - xs[:, None])
    candidate = grid & (dist_to_curve > MARGIN) & (~mask)
    cand_rows, cand_cols = np.where(candidate)
    if len(cand_rows) > N_SYNTH:
        pick = rng.choice(len(cand_rows), size=N_SYNTH, replace=False)
        cand_rows, cand_cols = cand_rows[pick], cand_cols[pick]
    n_synth = len(cand_rows)

    signed = cand_cols - xs[cand_rows]
    is_high_side = (signed <= 0) if high_left else (signed > 0)
    dist_high = np.where(is_high_side, np.abs(signed), np.nan)
    dist_low = np.where(~is_high_side, np.abs(signed), np.nan)
    invented = np.where(
        is_high_side,
        np.clip(_relax_func(np.abs(signed), *high_fit), 0, 1),
        np.clip(_relax_func(np.abs(signed), *low_fit), 0, 1),
    )

    composite = gt.copy()
    composite[cand_rows, cand_cols] = invented
    augmented_mask = mask.copy()
    augmented_mask[cand_rows, cand_cols] = True

    out_dir = os.path.join(MASKS_DIR, "g%s" % g, "mf%s" % mf, "synth_%s" % family)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(composite, 0, 1) * 255).astype(np.uint8), mode="L").save(
        os.path.join(out_dir, "image_composite.png"))
    np.save(os.path.join(out_dir, "mask_augmented.npy"), augmented_mask)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ry, rx = np.where(mask)
    axs[0].scatter(rx, ry, s=6, c="red")
    axs[0].plot(xs, np.arange(H), color="lime", linewidth=1.0)
    axs[0].set_title("%d reales (rojo) + frontera estimada" % n_real)
    axs[1].imshow(composite, cmap="gray", vmin=0, vmax=1)
    axs[1].scatter(cand_cols, cand_rows, s=4, c="cyan")
    axs[1].set_title("%d sinteticos (cian), valor inventado" % n_synth)
    axs[2].imshow(np.abs(composite - gt), cmap="magma")
    axs[2].set_title("|inventado - real| (diagnostico, no se usa)")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("g=%s mf=%s %s -> synth (%d reales + %d sinteticos = %d total)" % (
        g, mf, family, n_real, n_synth, n_real + n_synth))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=130)
    plt.close(fig)

    err = np.abs(composite - gt)[cand_rows, cand_cols]
    print("Reales: %d   Sinteticos: %d   Total mascara aumentada: %d" % (
        n_real, n_synth, int(augmented_mask.sum())))
    print("Error del valor inventado vs. el real (solo diagnostico, MAE): %.4f  max=%.4f" % (
        err.mean(), err.max()))
    print("Escrito en", out_dir)


if __name__ == "__main__":
    main()
