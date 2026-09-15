#!/usr/bin/env python
# coding: utf-8
"""Reconstruccion de la frontera de fase SIN red neuronal.

Alternativa a `dip.restoration` para esta familia de imagenes (dos regiones
separadas por una frontera monotona, con un salto brusco justo en el borde),
pensada para el regimen de MUY pocos puntos donde DIP arrastra (alucina
dientes de sierra en frentes abruptos, necesita tunear backtracking, etc. --
ver `dip.restoration`).

En vez de entrenar una red con un prior de "imagen natural", explota que el
objeto tiene pocos grados de libertad reales:

  1. Cada punto observado se clasifica en meseta alta/baja por el hueco mas
     grande en sus valores ordenados (son casi bimodales, no hace falta
     nada mas fino).
  2. Para cada punto, se busca su vecino mas cercano de la clase OPUESTA
     (KDTree) y se toma el punto medio entre ambos como una estimacion local
     de un punto SOBRE la frontera -- funciona aunque haya como mucho 1-2
     puntos por fila, porque no depende de tener ambas clases en la misma
     fila.
  3. Esas estimaciones se agrupan por fila (bins) y se interpola
     monotonamente entre ellas -- NO se fuerza un numero fijo de tramos
     rectos: la curva real puede ponerse casi vertical cerca de una esquina
     (una curvatura extra que un ajuste de 2 rectas no capturaba). Fuera del
     rango de filas con evidencia se extiende plana, nunca se extrapola una
     pendiente a ciegas.
  4. Cada lado de la curva NO es una meseta plana -- satura suavemente lejos
     de la frontera (se ve directo en `soc.npy`: el valor sigue subiendo/
     bajando varias filas despues del salto). Se ajusta una relajacion
     exponencial `C + A*exp(-d/L)` por lado (`d` = distancia a la frontera)
     a los mismos puntos clasificados, y se redibuja la imagen completa con
     esos dos campos + el salto angosto en la curva ajustada.

Determinista, sin entrenamiento, sin backtracking que tunear.

Uso:
    ./venv/bin/python -m curvefit.reconstruct <g> <mf> <family>
    ./venv/bin/python -m curvefit.reconstruct -4.0 0.980 grid

Lee:
    results/phase_diagram_g/g<g>/soc.npy                     (ground truth denso)
    data/restoration/masks_g/g<g>/mf<mf>/mask_<family>.npy   (puntos observados)
Escribe en results/curvefit/g<g>/mf<mf>/<family>/:
    reconstructed.png, comparison.png
e imprime PSNR/SSIM/MAE.

Variables de entorno (opcionales):
    TRANSITION_WIDTH   ancho (px) de la transicion redibujada (def: 8.0)
    CURVE_BANDWIDTH    ancho (filas) del nucleo gaussiano que suaviza la
                       curva de frontera (def: 2.0; mas alto = mas suave
                       pero redondea el codo)
"""
from __future__ import print_function

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402

TRANSITION_WIDTH = float(os.environ.get("TRANSITION_WIDTH", "8.0"))
CURVE_BANDWIDTH = float(os.environ.get("CURVE_BANDWIDTH", "2.0"))

PHASE_DIAGRAM_DIR = "results/phase_diagram_g"
MASKS_DIR = "data/restoration/masks_g"
OUT_ROOT = "results/curvefit"


def load_ground_truth(g):
    """soc.npy se guarda en orden de fila crudo (xi ascendente), pero
    `dip.phase_diagram` guarda el PNG que usa DIP con `np.flipud` (para que
    se vea con origin='lower' al abrirlo como imagen), y las mascaras
    `frontier`/`frontier_mix`/`spread` de `dip.frontier_mask` se calculan
    sobre esa MISMA version volteada. Hay que voltear soc.npy del mismo modo
    para que sus filas coincidan con las de la mascara y con `restored.png`
    de DIP -- si no, para esas 3 familias los puntos "cerca de la frontera"
    quedan en realidad cerca de la frontera ESPEJADA."""
    path = os.path.join(PHASE_DIAGRAM_DIR, "g%s" % g, "soc.npy")
    return np.flipud(np.load(path)).astype(np.float32).copy()


def find_level_threshold(vals):
    """Umbral entre las dos mesetas: el hueco mas grande en los valores
    observados ordenados. Bimodal por construccion (dos plateaus planos)."""
    sv = np.sort(vals)
    if len(sv) < 2:
        return float(np.median(vals))
    gaps = np.diff(sv)
    k = int(np.argmax(gaps))
    return float((sv[k] + sv[k + 1]) / 2.0)


def boundary_points(mask, gt, max_dist_factor=2.0):
    """Estima puntos sobre la frontera emparejando cada punto observado con
    su vecino mas cercano de la clase opuesta (no requiere ambas clases en
    la misma fila). Un punto lejos de la frontera tiene su vecino opuesto
    mas cercano a varias celdas de distancia -- ese par no dice nada sobre
    donde pasa la frontera ahi, asi que se DESCARTA (no solo se le baja el
    peso): el umbral es `max_dist_factor` veces el percentil 25 de todas las
    distancias, una estimacion robusta del espaciado real entre puntos
    vecinos de la mascara. Devuelve (filas, cols, pesos, umbral, high_rc, low_rc)."""
    rows, cols = np.where(mask)
    vals = gt[rows, cols]
    thresh = find_level_threshold(vals)
    is_high = vals >= thresh

    high_rc = np.column_stack([rows[is_high], cols[is_high]]).astype(float)
    low_rc = np.column_stack([rows[~is_high], cols[~is_high]]).astype(float)
    if len(high_rc) == 0 or len(low_rc) == 0:
        raise ValueError(
            "la mascara solo observa una meseta -- no se puede ubicar la frontera"
        )

    tree_high = cKDTree(high_rc)
    tree_low = cKDTree(low_rc)
    d_h, idx_h = tree_low.query(high_rc)
    d_l, idx_l = tree_high.query(low_rc)

    max_dist = max_dist_factor * float(np.percentile(np.concatenate([d_h, d_l]), 25))
    keep_h = d_h <= max_dist
    keep_l = d_l <= max_dist

    mid_rc = np.concatenate([
        (high_rc[keep_h] + low_rc[idx_h[keep_h]]) / 2.0,
        (low_rc[keep_l] + high_rc[idx_l[keep_l]]) / 2.0,
    ])
    weights = np.concatenate([1.0 / (1.0 + d_h[keep_h]), 1.0 / (1.0 + d_l[keep_l])])
    return mid_rc[:, 0], mid_rc[:, 1], weights, thresh, high_rc, low_rc


def fit_boundary_curve(rows, cols, weights, H, bandwidth=8.0):
    """Curva de frontera por regresion de nucleo (Nadaraya-Watson, gaussiano
    de ancho `bandwidth` filas) sobre las estimaciones de punto-de-frontera
    -- en vez de forzar un numero fijo de tramos rectos, o interpolar recto
    entre bins (demasiado ruidoso: cada estimacion es un unico par de vecinos
    mas cercanos, no una medicion limpia). El promedio ponderado por nucleo
    cancela ese ruido igual que ya se hacia en `smooth_fill`/`canonical_profile`
    de `analysis/correct_frontier_line.py`, pero acomodandose a filas con
    densidad de puntos despareja en vez de asumir una fila por muestra.
    Fuera de [rows.min(), rows.max()] no hay evidencia de cruce cerca (filas
    enteras de una sola clase) -- se extiende PLANA, no se extrapola.
    Devuelve una funcion fila -> columna.
    """
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    weights = np.asarray(weights, dtype=float)
    lo, hi = rows.min(), rows.max()

    def curve(rows_q):
        rq = np.clip(np.asarray(rows_q, dtype=float), lo, hi)
        d = (rq[:, None] - rows[None, :]) / bandwidth
        k = weights[None, :] * np.exp(-0.5 * d * d)
        wsum = k.sum(axis=1)
        wsum = np.where(wsum > 1e-9, wsum, 1.0)
        return (k * cols[None, :]).sum(axis=1) / wsum

    return curve


def _exp_safe(x):
    return np.exp(np.clip(x, -50, 50))


def _relax_func(d, C, A, L):
    """Relajacion hacia la asintota C, con salto A en d=0 y escala L."""
    return C + A * _exp_safe(-d / L)


def fit_relaxation(dist, vals):
    """Ajusta val(d) = C + A*exp(-d/L) por lado (d = distancia a la frontera,
    hacia adentro del lado en cuestion). Cada meseta NO es plana: satura
    suavemente lejos de la frontera y pega un salto justo en d=0 -- esto se
    ve directo en soc.npy (ej. g=-4.0: col fija, val pasa de ~0.91 a ~0.9996
    en pocas filas y despues se aplana). Devuelve (C, A, L); si el ajuste no
    converge (muy pocos puntos, todos a la misma distancia), cae a un
    relleno plano (A=0, L=1) con la mediana como C.
    """
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
            bounds=([0.0, -1.0, 1e-3], [1.0, 1.0, 500.0]),
            maxfev=5000,
        )
        return float(C), float(A), float(L)
    except RuntimeError:
        return C0, A0, L0


def redraw(shape, xs_by_row, high_fit, low_fit, width, high_left):
    """Cada lado de xs_by_row[fila] se rellena con SU relajacion ajustada
    (`fit_relaxation`), no con un escalar -- el salto brusco queda solo en
    la transicion angosta de `width` px alrededor de la frontera."""
    H, W = shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    signed = cols - xs_by_row[:, None]  # >0 hacia la derecha de la frontera

    dist_high = -signed if high_left else signed
    dist_low = signed if high_left else -signed
    field_high = np.clip(_relax_func(dist_high, *high_fit), 0, 1)
    field_low = np.clip(_relax_func(dist_low, *low_fit), 0, 1)

    c = xs_by_row[:, None]
    t = np.clip((cols - (c - width / 2.0)) / max(width, 1e-6), 0, 1)
    out = (field_high * (1 - t) + field_low * t) if high_left else (field_low * (1 - t) + field_high * t)
    return out.astype(np.float32)


def reconstruct(g, mf, family, width=None, bandwidth=None):
    bandwidth = CURVE_BANDWIDTH if bandwidth is None else bandwidth
    width = TRANSITION_WIDTH if width is None else width

    gt = load_ground_truth(g)
    mask = np.load(os.path.join(MASKS_DIR, "g%s" % g, "mf%s" % mf, "mask_%s.npy" % family))
    H, W = gt.shape

    mid_rows, mid_cols, weights, thresh, high_rc, low_rc = boundary_points(mask, gt)
    curve = fit_boundary_curve(mid_rows, mid_cols, weights, H, bandwidth)

    rows_full = np.arange(H)
    xs = np.clip(curve(rows_full), 0, W - 1)

    obs_vals = gt[mask]
    high_val = float(np.median(obs_vals[obs_vals >= thresh]))
    low_val = float(np.median(obs_vals[obs_vals < thresh]))
    high_left = bool(high_rc[:, 1].mean() <= low_rc[:, 1].mean())

    xs_at_high = xs[high_rc[:, 0].astype(int)]
    xs_at_low = xs[low_rc[:, 0].astype(int)]
    dist_high_obs = (xs_at_high - high_rc[:, 1]) if high_left else (high_rc[:, 1] - xs_at_high)
    dist_low_obs = (low_rc[:, 1] - xs_at_low) if high_left else (xs_at_low - low_rc[:, 1])
    high_fit = fit_relaxation(dist_high_obs, gt[high_rc[:, 0].astype(int), high_rc[:, 1].astype(int)])
    low_fit = fit_relaxation(dist_low_obs, gt[low_rc[:, 0].astype(int), low_rc[:, 1].astype(int)])

    rec = redraw((H, W), xs, high_fit, low_fit, width, high_left)
    rec = np.clip(rec, 0, 1)

    psnr = dip_metrics.psnr(gt[None], rec[None])
    ssim = dip_metrics.ssim(gt[None], rec[None], 1)
    mae = dip_metrics.mae(gt[None], rec[None])

    return {
        "g": g, "mf": mf, "family": family,
        "n_obs": int(mask.sum()), "threshold": thresh,
        "high": high_val, "low": low_val, "high_left": high_left,
        "high_fit": high_fit, "low_fit": low_fit,
        "psnr": float(psnr), "ssim": float(ssim), "mae": mae,
        "gt": gt, "mask": mask, "rec": rec, "xs": xs,
        "mid_rows": mid_rows, "mid_cols": mid_cols,
        "high_rc": high_rc, "low_rc": low_rc,
    }


def save_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gt, rec, mask = result["gt"], result["rec"], result["mask"]
    H, W = gt.shape
    y = np.arange(H)

    plt.imsave(os.path.join(out_dir, "reconstructed.png"), rec, cmap="gray", vmin=0, vmax=1)

    fig, axs = plt.subplots(1, 3, figsize=(11, 4))
    axs[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Original (soc.npy)")

    axs[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[1].scatter(result["high_rc"][:, 1], result["high_rc"][:, 0], s=6, c="#d62728", label="alto")
    axs[1].scatter(result["low_rc"][:, 1], result["low_rc"][:, 0], s=6, c="#1f77b4", label="bajo")
    axs[1].legend(fontsize=7, loc="lower right")
    axs[1].set_title("%d puntos observados" % result["n_obs"])

    axs[2].imshow(rec, cmap="gray", vmin=0, vmax=1)
    axs[2].plot(result["xs"], y, color="lime", linewidth=1.2)
    axs[2].set_title("Curve-fit (PSNR %.1f, SSIM %.3f)" % (result["psnr"], result["ssim"]))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("g=%s  mf=%s  %s  (%d pts, sin red neuronal)" % (
        result["g"], result["mf"], result["family"], result["n_obs"]))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=130)
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    g, mf, family = sys.argv[1], sys.argv[2], sys.argv[3]

    result = reconstruct(g, mf, family)
    out_dir = os.path.join(OUT_ROOT, "g%s" % g, "mf%s" % mf, family)
    save_outputs(result, out_dir)

    print("Umbral meseta alta/baja: %.4f  (mediana alta=%.3f, baja=%.3f, alta a la %s)" % (
        result["threshold"], result["high"], result["low"],
        "izquierda" if result["high_left"] else "derecha"))
    print("Relajacion alta  C=%.4f A=%.4f L=%.2f" % result["high_fit"])
    print("Relajacion baja  C=%.4f A=%.4f L=%.2f" % result["low_fit"])
    print("PSNR: %.2f dB   SSIM: %.4f   MAE: %.4f  (%d puntos observados)" % (
        result["psnr"], result["ssim"], result["mae"], result["n_obs"]))
    print("Escrito en", out_dir)


if __name__ == "__main__":
    main()
