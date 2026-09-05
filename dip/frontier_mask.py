#!/usr/bin/env python
# coding: utf-8
"""Genera mascaras de muestreo NO uniformes para el inpainting DIP, pensadas para
probar cuanto pesa tener (o no tener) puntos observados sobre la frontera de fase
del mapa SoC_max(log Xi, log l).

Para cada fraccion observada de OBS_FRACS produce 4 mascaras con el mismo N:
  - frontier     : casi todos los puntos caen sobre la frontera (alto |grad SoC|).
  - frontier_mix : la mayoria sobre la frontera, pero una fraccion MIX_FRAC
                   repartida de forma uniforme fuera de ella.
  - spread       : los puntos evitan la frontera (se reparten por las mesetas).
  - uniform      : baseline Bernoulli uniforme (para comparar).

Las mascaras de una misma familia son *anidadas*: los N puntos de una fraccion
chica son un subconjunto de los de la fraccion mas grande (se sortea una vez el N
maximo y se toman prefijos). Asi el barrido en % es limpio.

Estructura de salida (espeja a slurm/dip_sweep.slurm -> mf<frac>/):
    data/restoration/masks/mf<hidden>/mask_<name>.npy   (+ .png)
    data/restoration/masks/mf<hidden>/preview.png

Cada mascara se genera al tamano del PNG de entrada (recortado a multiplo de 64);
dip.restoration la lee via MASK_PATH y le aplica el mismo reflection-pad(+1) que a
la imagen.

Se ejecuta como modulo desde la raiz del repo:
    ./venv/bin/python -m dip.frontier_mask

Variables de entorno (todas opcionales):
    IMAGE_PATH     PNG de entrada (input DIP)     (def: ./data/restoration/phase_diagram_sim_128.png)
    SOC_NPY        grilla SoC cruda para definir la frontera
                                                  (def: ./archive/phase_diagram/phase_diagram_soc_128.npy)
    OUT_DIR        carpeta de salida              (def: ./data/restoration/masks)
    OBS_FRACS      fracciones OBSERVADAS, separadas por espacio
                                                  (def: "0.02 0.01 0.005 0.003 0.002 0.001"
                                                   -> MASK_FRAC 0.98 .. 0.999)
    SEED          semilla del muestreo            (def: 42)
    SMOOTH_SIGMA   sigma del suavizado del |grad| (def: 1.0)
    EDGE_GAMMA     exponente del peso de frontera (def: 3.0)
    MIX_FRAC       en frontier_mix, peso relativo del termino uniforme (def: 0.15)
"""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from utils.common_utils import crop_image

IMAGE_PATH = os.environ.get("IMAGE_PATH", "./data/restoration/phase_diagram_sim_128.png")
SOC_NPY = os.environ.get("SOC_NPY", "./archive/phase_diagram/phase_diagram_soc_128.npy")
OUT_DIR = os.environ.get("OUT_DIR", "./data/restoration/masks")
OBS_FRACS = [
    float(x)
    for x in os.environ.get("OBS_FRACS", "0.02 0.01 0.005 0.003 0.002 0.001").split()
]
SEED = int(os.environ.get("SEED", "42"))
SMOOTH_SIGMA = float(os.environ.get("SMOOTH_SIGMA", "1.0"))
EDGE_GAMMA = float(os.environ.get("EDGE_GAMMA", "3.0"))
MIX_FRAC = float(os.environ.get("MIX_FRAC", "0.15"))

os.makedirs(OUT_DIR, exist_ok=True)

# --- Imagen tal cual la ve DIP: PNG -> recorte a multiplo de 64 -> [0,1] --------
img_pil = Image.open(IMAGE_PATH).convert("L")
img_pil = crop_image(img_pil, 64)
img = np.asarray(img_pil, dtype=np.float32) / 255.0
H, W = img.shape
print("Imagen DIP:", IMAGE_PATH, "->", img.shape)

# --- Campo de "frontera" = |grad SoC| suavizado y normalizado a [0,1] ----------
# Se usa la grilla cruda si esta disponible (menos escalonada que el PNG de 8 bit),
# reorientada a la convencion del PNG (fila 0 = xi mas alto).
if os.path.exists(SOC_NPY):
    soc = np.load(SOC_NPY).astype(np.float32)
    soc = np.flipud(soc)  # phase_diagram.py guarda xi ascendente; el PNG va al reves
    if soc.shape != (H, W):
        soc = np.asarray(
            Image.fromarray((np.clip(soc, 0, 1) * 255).astype(np.uint8)).resize(
                (W, H), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    print("Frontera desde grilla cruda:", SOC_NPY)
else:
    soc = img
    print("Frontera desde el PNG (no se encontro SOC_NPY)")

gy, gx = np.gradient(ndi.gaussian_filter(soc, SMOOTH_SIGMA))
gmag = np.hypot(gx, gy)
gmag = ndi.gaussian_filter(gmag, SMOOTH_SIGMA)
edge = gmag / max(gmag.max(), 1e-12)  # [0,1], 1 = frontera
flat_edge = edge.reshape(-1)

# pesos de sorteo (densidades de probabilidad, una por familia de mascara)
w_edge = flat_edge ** EDGE_GAMMA + 1e-4
w_edge = w_edge / w_edge.sum()
w_unif = np.full_like(w_edge, 1.0 / w_edge.size)
w_spread = (1.0 - flat_edge) ** EDGE_GAMMA + 1e-4
w_spread = w_spread / w_spread.sum()

WEIGHTS = {
    "frontier": w_edge,
    "frontier_mix": (1.0 - MIX_FRAC) * w_edge + MIX_FRAC * w_unif,
    "spread": w_spread,
    "uniform": w_unif,
}
ORDER = ["frontier", "frontier_mix", "spread", "uniform"]

obs_sorted = sorted(OBS_FRACS, reverse=True)
n_max = int(round(obs_sorted[0] * H * W))
print(
    "Fracciones observadas:", obs_sorted,
    "-> N =", [int(round(o * H * W)) for o in obs_sorted], "(de %dx%d)" % (H, W),
)

# un orden de sorteo por familia (N maximo); las fracciones chicas son prefijos
full_order = {}
for k, name in enumerate(ORDER):
    rng = np.random.default_rng([SEED, k])
    full_order[name] = rng.choice(H * W, size=n_max, replace=False, p=WEIGHTS[name])

xs_lin = np.linspace(-4, 2, W)
ys_lin = np.linspace(2, -4, H)
edge_lvl = np.quantile(edge, 0.90)

for obs in obs_sorted:
    mf = 1.0 - obs
    N = int(round(obs * H * W))
    sub = os.path.join(OUT_DIR, "mf%.3f" % mf)
    os.makedirs(sub, exist_ok=True)

    masks = {}
    print("mf%.3f  (obs %.2f%%, N=%d)" % (mf, 100 * obs, N))
    for name in ORDER:
        flat = np.zeros(H * W, dtype=bool)
        flat[full_order[name][:N]] = True
        m = flat.reshape(H, W)
        masks[name] = m
        np.save(os.path.join(sub, "mask_%s.npy" % name), m)
        Image.fromarray((m * 255).astype(np.uint8), mode="L").save(
            os.path.join(sub, "mask_%s.png" % name)
        )
        on_edge = float((m & (edge > 0.30)).sum()) / max(m.sum(), 1)
        print("    %-12s %4d pts  %.0f%% sobre frontera" % (name, int(m.sum()), 100 * on_edge))

    fig, axs = plt.subplots(2, 2, figsize=(11, 10.5))
    for ax, name in zip(axs.flat, ORDER):
        ax.imshow(soc, cmap="viridis", vmin=0, vmax=1, origin="upper",
                  extent=[-4, 2, -4, 2], aspect="auto")
        ax.contour(xs_lin, ys_lin, edge, levels=[edge_lvl],
                   colors="white", linewidths=0.8, alpha=0.7)
        ys, xs = np.nonzero(masks[name])
        ax.scatter(-4 + (xs + 0.5) / W * 6, 2 - (ys + 0.5) / H * 6,
                   s=16, c="red", edgecolors="k", linewidths=0.3)
        ax.set_title("%s  (%d pts)" % (name, int(masks[name].sum())))
        ax.set_xlabel(r"$\log(\ell)$")
        ax.set_ylabel(r"$\log(\Xi)$")
    fig.suptitle(
        "mf%.3f  -  N=%d obs (%.2f%%)  -  linea blanca = frontera (p90 |grad|)"
        % (mf, N, 100 * obs)
    )
    fig.tight_layout()
    fig.savefig(os.path.join(sub, "preview.png"), dpi=120)
    plt.close(fig)

print("Listo. Mascaras en:", OUT_DIR, "(subcarpetas mf*)")
