#!/usr/bin/env python
# coding: utf-8
"""Metodologia TOTALMENTE distinta a `dip.restoration`/`curvefit.reconstruct`:
en vez de reconstruir la IMAGEN a partir de puntos dispersos, se infiere el
PARAMETRO FISICO `g` (interaccion de Frumkin) que la genero, y se regenera el
mapa completo corriendo de nuevo el modelo continuo (`galpynostatic`) con ese
`g` -- no hay "imagen" que rellenar, hay un problema inverso de 1 parametro.

Por que puede andar con MUY pocos puntos (<60): estos mapas no son imagenes
arbitrarias, son la salida de `dip.phase_diagram` (single-particle model +
Butler-Volmer), que tiene esencialmente UN grado de libertad libre en el
barrido de `g` negativo (`vcut` queda fijo en -0.15). Encontrar 1 numero real
que mejor explica ~60 muestras (ξ, ℓ, SoC) es un problema muchisimo mejor
condicionado que "rellenar" una grilla de 128x128 = 16384 pixeles -- y una
vez encontrado `g`, el mapa completo se obtiene corriendo el modelo (exacto,
no interpolado ni alucinado), no aproximandolo.

Limitacion sabida: solo tiene sentido si el objeto que se quiere reconstruir
de verdad viene de esta familia de 1 parametro (cierto para las imagenes
sinteticas de este barrido, generadas por `dip.phase_diagram` variando `g`;
para datos KMC reales el modelo continuo es una APROXIMACION, asi que esto
daria el mejor `g` que explica los puntos, no necesariamente reconstruye
desviaciones reales del continuo -- ver docstring de mas abajo / memoria de
proyecto).

`galpynostatic.simulation.GalvanostaticMap` solo evalua grillas rectangulares
(producto cartesiano de un array de xi y un array de ell), no puntos sueltos
arbitrarios -- pero corriendola con num_xi=1 y logL_ pisado a mano con los
valores de ell EXACTOS de una fila (bypaseando el linspace interno) evalua
el modelo en cualquier punto disperso, agrupando por fila para pagar el
overhead fijo por corrida una sola vez por fila en vez de por punto
(~1.2s fijo + ~0.43s/punto, medido).

Uso:
    ./venv/bin/python -m physics_fit.calibrate <g_real> <n_puntos>
    ./venv/bin/python -m physics_fit.calibrate -4.0 60
"""
from __future__ import print_function

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import galpynostatic.simulation as gpsim

VCUT = -0.15  # fijo, igual que en el barrido (solo g varia)
GRID_SIZE = 1000
TIME_STEPS = 100_000
H = W = 128
LOGXI_LOW, LOGXI_HIGH = -4.0, 2.0
LOGELL_LOW, LOGELL_HIGH = -4.0, 2.0

PHASE_DIAGRAM_DIR = "results/phase_diagram_g"
OUT_ROOT = "results/physics_fit"


def xi_axis(H=H):
    # fila 0 = xi mas alto, igual que el PNG/mascaras de dip.phase_diagram (flipud)
    return np.linspace(LOGXI_HIGH, LOGXI_LOW, H)


def ell_axis(W=W):
    return np.linspace(LOGELL_LOW, LOGELL_HIGH, W)


def load_ground_truth(g):
    """Misma convencion volteada que dip.frontier_mask / curvefit.reconstruct
    (ver memoria de proyecto): dip.phase_diagram guarda soc.npy crudo pero el
    PNG que usa DIP esta con np.flipud."""
    path = os.path.join(PHASE_DIAGRAM_DIR, "g%s" % g, "soc.npy")
    return np.flipud(np.load(path)).astype(np.float32).copy()


def eval_row(xi_val, ell_vals, g, vcut=VCUT, grid_size=GRID_SIZE, time_steps=TIME_STEPS):
    """SoC del modelo continuo en un xi fijo y varios ell arbitrarios (no
    necesariamente equiespaciados) -- pisa logL_/logxi_ despues de construir
    para evitar el linspace interno, que solo genera grillas equiespaciadas."""
    n = len(ell_vals)
    gm = gpsim.GalvanostaticMap(
        vcut=vcut, g=g,
        logxi_lle=xi_val, logxi_ule=xi_val, num_xi=1,
        logell_lle=0.0, logell_ule=1.0, num_ell=n,
        grid_size=grid_size, time_steps=time_steps, nthreads=-1,
    )
    gm.logL_ = np.asarray(ell_vals, dtype=np.float64)
    gm.logxi_ = np.asarray([xi_val], dtype=np.float64)
    gm.run()
    df = gm.map_dataframe
    lookup = dict(zip(df["ell"].to_numpy(), df["SOC"].to_numpy()))
    return np.array([lookup[e] for e in ell_vals])


def evaluate_model_at_points(rows, cols, g, vcut=VCUT, grid_size=GRID_SIZE, time_steps=TIME_STEPS,
                              H=H, W=W):
    """SoC del modelo continuo en un set disperso de (fila, columna) de una
    imagen HxW -- agrupa por fila (mismo xi) para minimizar cantidad de
    corridas de `GalvanostaticMap`. H/W deben coincidir con la resolucion en
    la que se tomaron `rows`/`cols` (si se muestrea en una grilla mas chica
    que la 128x128 default, hay que pasarlos explicitamente -- si no, el eje
    fisico xi/ell queda mal escalado)."""
    xi_ax, ell_ax = xi_axis(H), ell_axis(W)
    xi_vals = xi_ax[rows]
    ell_vals = ell_ax[cols]
    result = np.empty(len(rows), dtype=np.float64)
    for xv in np.unique(xi_vals):
        idx = np.where(xi_vals == xv)[0]
        result[idx] = eval_row(xv, ell_vals[idx], g, vcut, grid_size, time_steps)
    return result


def sample_points(H, W, n, seed=0):
    """N puntos en grilla 2D lo mas pareja posible (misma familia 'grid' que
    ya se uso para DIP/curve-fit) -- deterministico, sin azar."""
    side = max(1, int(round(np.sqrt(n))))
    rows = np.linspace(0, H - 1, side).round().astype(int)
    cols = np.linspace(0, W - 1, side).round().astype(int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel()


def fit_g(rows, cols, observed, bounds=(-6.0, 1.0), grid_size=GRID_SIZE, time_steps=TIME_STEPS,
          H=H, W=W):
    """Ajusta el escalar `g` minimizando el error cuadratico entre el modelo
    continuo evaluado en los mismos puntos dispersos y los valores
    observados -- problema inverso de 1 parametro, no reconstruccion de
    imagen. H/W: resolucion de la grilla en la que estan `rows`/`cols`."""
    n_eval = [0]

    def loss(g):
        n_eval[0] += 1
        pred = evaluate_model_at_points(rows, cols, g, grid_size=grid_size, time_steps=time_steps, H=H, W=W)
        err = float(np.mean((pred - observed) ** 2))
        print("  eval #%d  g=%+.4f  MSE=%.6f" % (n_eval[0], g, err))
        return err

    res = minimize_scalar(loss, bounds=bounds, method="bounded",
                           options={"xatol": 0.05})
    return float(res.x), n_eval[0]


def regenerate_full_map(g, vcut=VCUT, grid_size=GRID_SIZE, time_steps=TIME_STEPS, H=H, W=W):
    """Corre el modelo completo (grilla HxW) con el `g` ajustado -- esto NO
    es una imagen 'reconstruida' por interpolacion, es la salida exacta del
    modelo fisico para ese parametro."""
    gm = gpsim.GalvanostaticMap(
        vcut=vcut, g=g,
        logxi_lle=LOGXI_HIGH, logxi_ule=LOGXI_LOW, num_xi=H,
        logell_lle=LOGELL_HIGH, logell_ule=LOGELL_LOW, num_ell=W,
        grid_size=grid_size, time_steps=time_steps, nthreads=-1,
    )
    gm.run()
    df = gm.map_dataframe
    pivot = df.pivot(index="xi", columns="ell", values="SOC").sort_index(axis=0).sort_index(axis=1)
    soc_grid = pivot.to_numpy(dtype=np.float32)
    return np.flipud(np.clip(soc_grid, 0, 1))  # misma convencion volteada


def main():
    g_real = sys.argv[1] if len(sys.argv) > 1 else "-4.0"
    n_pts = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    gt = load_ground_truth(g_real)
    rows, cols = sample_points(H, W, n_pts)
    observed = gt[rows, cols]
    print("g real=%s, %d puntos observados (grilla %dx%d)" % (g_real, len(rows), H, W))

    t0 = time.time()
    g_hat, n_eval = fit_g(rows, cols, observed)
    fit_time = time.time() - t0
    print("\ng_hat=%.4f (real=%s), %d evaluaciones, %.1f s" % (g_hat, g_real, n_eval, fit_time))

    t0 = time.time()
    rec = regenerate_full_map(g_hat)
    gen_time = time.time() - t0
    print("Mapa completo regenerado en %.1f s" % gen_time)

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from dip import metrics as dip_metrics
    psnr = dip_metrics.psnr(gt[None], rec[None])
    ssim = dip_metrics.ssim(gt[None], rec[None], 1)
    mae = dip_metrics.mae(gt[None], rec[None])
    print("PSNR: %.2f dB   SSIM: %.4f   MAE: %.4f" % (psnr, ssim, mae))

    out_dir = os.path.join(OUT_ROOT, "g%s" % g_real, "n%d" % n_pts)
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(11, 4))
    axs[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[0].scatter(cols, rows, s=8, c="red")
    axs[0].set_title("Original + %d puntos" % len(rows))
    axs[1].imshow(rec, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Modelo con g_hat=%.3f" % g_hat)
    axs[2].imshow(np.abs(gt - rec), cmap="inferno", vmin=0, vmax=1)
    axs[2].set_title("|error|")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("g real=%s -> g_hat=%.3f  (PSNR %.1f, SSIM %.3f, %d pts, %d evals, %.0fs)" % (
        g_real, g_hat, psnr, ssim, len(rows), n_eval, fit_time))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=130)
    plt.close(fig)
    print("Escrito", os.path.join(out_dir, "comparison.png"))


if __name__ == "__main__":
    main()
