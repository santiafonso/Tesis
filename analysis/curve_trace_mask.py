#!/usr/bin/env python
# coding: utf-8
"""Mascara "curve_trace": reparte los puntos observados de forma PAREJA por
longitud de arco a lo largo de la frontera real (extraida del soc.npy denso,
mismo metodo que dip.frontier_mask: |grad SoC| suavizado, contorno p90), mas
un puñado fijo en las mesetas para anclar los niveles. A diferencia de
`frontier`/`frontier_mix` (sorteo aleatorio ponderado por el gradiente), esto
es determinista: cubre la curva entera con espaciado minimo garantizado, en
vez de dejar que el azar amontone o deje huecos.

Sigue siendo DIP el que reconstruye -- esto solo cambia que puntos se le dan.
Pensado para el regimen de MUY pocos puntos (<50), donde el muestreo al azar
empieza a fallar por varianza.

Uso:
    G=-4.0 N=48 N_PLATEAU=16 ./venv/bin/python -m analysis.curve_trace_mask
"""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage import measure

G = os.environ.get("G", "-4.0")
N = int(os.environ.get("N", "48"))
N_PLATEAU = int(os.environ.get("N_PLATEAU", "16"))  # del total N, cuantos van a las mesetas
PHASE_G_DIR = os.environ.get("PHASE_G_DIR", "results/phase_diagram_g")
OUT_DIR = os.environ.get("OUT_DIR", "data/restoration/masks_active")
SMOOTH_SIGMA = 1.0
EDGE_QUANTILE = 0.90
PLATEAU_QUANTILE = 0.10  # edge por debajo de este percentil = "meseta profunda"

N_FRONTIER = N - N_PLATEAU


def load_soc(g):
    soc_path = os.path.join(PHASE_G_DIR, "g%s" % g, "soc.npy")
    if os.path.exists(soc_path):
        soc = np.load(soc_path).astype(np.float32)
        return np.flipud(soc)
    png_path = os.path.join(PHASE_G_DIR, "g%s" % g, "sim_128.png")
    return np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0


def edge_field(soc):
    gy, gx = np.gradient(ndi.gaussian_filter(soc, SMOOTH_SIGMA))
    gmag = np.hypot(gx, gy)
    gmag = ndi.gaussian_filter(gmag, SMOOTH_SIGMA)
    return gmag / max(gmag.max(), 1e-12)


def frontier_points_even(edge, lvl, n_pts):
    """N puntos parejos por longitud de arco a lo largo del contorno edge==lvl."""
    contours = measure.find_contours(edge, level=lvl)
    if not contours:
        return np.zeros((0, 2))
    # concatenar todos los segmentos, ponderando por su longitud de arco propia
    all_pts = []
    lengths = []
    for c in contours:
        seg_len = np.sum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))
        all_pts.append(c)
        lengths.append(seg_len)
    total_len = sum(lengths)
    pts = []
    for c, seg_len in zip(all_pts, lengths):
        n_seg = max(1, int(round(n_pts * seg_len / total_len)))
        d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))])
        targets = np.linspace(0, d[-1], n_seg, endpoint=False) + (d[-1] / n_seg) / 2
        idx = np.searchsorted(d, targets)
        idx = np.clip(idx, 0, len(c) - 1)
        pts.append(c[idx])
    return np.concatenate(pts, axis=0)[:n_pts]


def _even_subgrid_in_region(region_mask, n_pts):
    """n_pts puntos por grilla regular, restringidos a region_mask (bool 2D)."""
    H, W = region_mask.shape
    if n_pts <= 0 or not region_mask.any():
        return np.zeros((0, 2))
    step = max(1, int(round(np.sqrt(H * W / n_pts))))
    while step >= 1:
        ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
        cand = np.stack([ys.ravel(), xs.ravel()], axis=1)
        cand = cand[region_mask[cand[:, 0], cand[:, 1]]]
        if len(cand) >= n_pts:
            idx = np.linspace(0, len(cand) - 1, n_pts).astype(int)
            return cand[idx].astype(float)
        step -= 1
    return cand.astype(float)


def plateau_points_even(soc, n_pts):
    """n_pts puntos repartidos MITAD Y MITAD entre las dos mesetas, definidas
    por VALOR (soc cerca de su minimo / cerca de su maximo), no por gradiente:
    una de las dos mesetas del modelo se satura de verdad (gradiente exacto
    0), la otra solo se acerca asintoticamente (gradiente nunca es exacto 0),
    asi que un umbral de gradiente deja esa meseta entera sin puntos."""
    lo_val = np.quantile(soc, PLATEAU_QUANTILE)
    hi_val = np.quantile(soc, 1 - PLATEAU_QUANTILE)
    low = soc <= lo_val
    high = soc >= hi_val
    n_lo = n_pts // 2
    n_hi = n_pts - n_lo
    pts = [_even_subgrid_in_region(low, n_lo), _even_subgrid_in_region(high, n_hi)]
    pts = [p for p in pts if len(p)]
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


def build_mask(g, n, n_plateau):
    soc = load_soc(g)
    H, W = soc.shape
    edge = edge_field(soc)
    lvl = np.quantile(edge, EDGE_QUANTILE)
    n_frontier = n - n_plateau
    fp = frontier_points_even(edge, lvl, n_frontier)
    pp = plateau_points_even(soc, n_plateau)
    pts = np.concatenate([fp, pp], axis=0) if len(fp) else pp
    rc = np.round(pts).astype(int)
    rc[:, 0] = np.clip(rc[:, 0], 0, H - 1)
    rc[:, 1] = np.clip(rc[:, 1], 0, W - 1)
    mask = np.zeros((H, W), dtype=bool)
    mask[rc[:, 0], rc[:, 1]] = True
    return mask, soc, edge, lvl


def main():
    mask, soc, edge, lvl = build_mask(G, N, N_PLATEAU)
    out = os.path.join(OUT_DIR, "g%s" % G)
    os.makedirs(out, exist_ok=True)
    n_real = int(mask.sum())
    np.save(os.path.join(out, "mask_curve_trace_N%d.npy" % N), mask)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
        os.path.join(out, "mask_curve_trace_N%d.png" % N)
    )
    print("g=%s  N pedido=%d  N real=%d (dedup por redondeo a pixel)" % (G, N, n_real))

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.imshow(soc, cmap="viridis", vmin=0, vmax=1)
    ax.contour(edge, levels=[lvl], colors="white", linewidths=0.8, alpha=0.6)
    ys, xs = np.nonzero(mask)
    ax.scatter(xs, ys, s=22, c="red", edgecolors="k", linewidths=0.3)
    ax.set_title("curve_trace  g=%s  N=%d (%d frontera + %d meseta)" % (G, n_real, N - N_PLATEAU, N_PLATEAU))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "preview_curve_trace_N%d.png" % N), dpi=130)
    plt.close(fig)
    print("Escrito", out)


if __name__ == "__main__":
    main()
