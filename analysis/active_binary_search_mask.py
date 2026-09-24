#!/usr/bin/env python
# coding: utf-8
"""Idea 15/9 (muestreo ACTIVO, la version realista de "inventar la meseta"):
en vez de sortear todos los puntos de una con un peso fijo (que necesita
conocer el campo denso de antemano -- no desplegable con KMC real), ubicar
la frontera con POCAS consultas ADAPTATIVAS, como si cada consulta fuera una
corrida real de KMC:

  1. Por una fila "sonda" (unas pocas, repartidas en la imagen), busqueda
     BINARIA horizontal: se pregunta el punto medio, se ve de que lado del
     cambio cae, se repite -- ~log2(W) consultas ubican el cruce en esa fila
     con precision de 1px, en vez de sortear puntos al azar esperando caer
     cerca.
  2. Todas las consultas del camino (no solo el cruce final) quedan como
     puntos reales "gratis" a distintas distancias del cruce -- sirven para
     ajustar la relajacion exponencial de la meseta (ver
     analysis/frontier_synth_augment.py, misma idea).
  3. Con los cruces de las pocas filas sonda se arma una curva de frontera
     (interpolacion), y el presupuesto de puntos reales que sobra se
     concentra PEGADO a esa curva ya encontrada -- no repartido uniforme ni
     con un peso asumido de antemano.
  4. Opcional: ademas de los puntos reales, "inventar" puntos sinteticos de
     meseta con la relajacion ajustada (mismo mecanismo que
     frontier_synth_augment, reusado aca).

Todo el mecanismo solo mira el `soc.npy` como si fuera el oraculo de una
consulta KMC puntual -- nunca lee el campo denso completo de antemano, asi
que es desplegable con KMC real (a diferencia de dip/frontier_mask.py, que
sí necesita el campo denso para las familias frontier/frontier_mix/spread).

Uso:
    ./venv/bin/python -m analysis.active_binary_search_mask -4.0

Escribe en data/restoration/masks_g/g<g>/active_search/:
    image_composite.png   mask_augmented.npy   preview.png
e imprime cuantas consultas (binarias + extra + sinteticas) se usaron.

Variables de entorno (opcionales):
    N_PROBES    filas sonda para la busqueda binaria (def: 20)
    N_STEPS     pasos de busqueda binaria por fila (def: 8 -> ~1px en W=128)
    N_EXTRA     puntos reales extra concentrados cerca de la curva ya
                encontrada (def: 150)
    N_SYNTH     puntos sinteticos (valor inventado, gratis) (def: 400)
    EXTRA_SPREAD  dispersion (px, std de la gaussiana) de los puntos extra
                  alrededor de la curva encontrada (def: 4.0)
    MARGIN      distancia minima (px) a la curva para poner un sintetico
                (def: 6.0)
"""
from __future__ import print_function

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.frontier_synth_augment import (  # noqa: E402
    _relax_func, fit_relaxation, load_ground_truth,
)

N_PROBES = int(os.environ.get("N_PROBES", "20"))
N_STEPS = int(os.environ.get("N_STEPS", "8"))
N_EXTRA = int(os.environ.get("N_EXTRA", "150"))
N_SYNTH = int(os.environ.get("N_SYNTH", "400"))
EXTRA_SPREAD = float(os.environ.get("EXTRA_SPREAD", "4.0"))
MARGIN = float(os.environ.get("MARGIN", "6.0"))
SEED = int(os.environ.get("SEED", "42"))

MASKS_DIR = "data/restoration/masks_g"


def binary_search_row(gt, row, threshold, n_steps):
    """Busqueda binaria horizontal en `row`: asume que gt[row] va de ALTO
    (col chica) a BAJO (col grande), como el resto del proyecto. Devuelve
    la lista de (col, valor) CONSULTADAS (todas, no solo la final) y la
    columna de cruce estimada."""
    W = gt.shape[1]
    lo, hi = 0, W - 1
    queried = []
    v_lo = gt[row, lo]
    v_hi = gt[row, hi]
    if (v_lo - threshold) * (v_hi - threshold) > 0:
        # toda la fila del mismo lado (meseta de punta a punta): 2 consultas
        # bastan para confirmarlo, no hace falta la busqueda completa.
        queried = [(lo, v_lo), (hi, v_hi)]
        return queried, None
    for _ in range(n_steps):
        mid = (lo + hi) // 2
        v = gt[row, mid]
        queried.append((mid, v))
        if v >= threshold:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1:
            break
    return queried, (lo + hi) / 2.0


def main():
    g = sys.argv[1] if len(sys.argv) > 1 else "-4.0"
    gt = load_ground_truth(g)
    H, W = gt.shape
    threshold = 0.5  # el campo esta en [0,1], meseta alta cerca de 1, baja cerca de 0

    probe_rows = np.linspace(0, H - 1, N_PROBES).astype(int)
    all_queries = []  # (row, col, val)
    crossings = []  # (row, col) solo donde se encontro cruce real
    for row in probe_rows:
        queried, cross_col = binary_search_row(gt, row, threshold, N_STEPS)
        for col, val in queried:
            all_queries.append((row, col, val))
        if cross_col is not None:
            crossings.append((row, cross_col))
    crossings = np.array(crossings, dtype=float)
    n_binary = len(all_queries)
    print("Busqueda binaria: %d filas sonda, %d consultas reales, %d cruces encontrados" % (
        N_PROBES, n_binary, len(crossings)))

    # --- curva de frontera: interpolar entre los cruces encontrados -------
    if len(crossings) >= 2:
        xs = np.interp(np.arange(H), crossings[:, 0], crossings[:, 1])
    else:
        xs = np.full(H, W / 2.0)

    # --- clasificar TODAS las consultas de la busqueda binaria en alto/bajo,
    # y ajustar la relajacion exponencial por lado (mismos puntos "gratis"
    # que ya se pagaron ubicando la frontera, no se gasta nada extra aca) --
    aq = np.array(all_queries)  # (n,3): row, col, val
    is_high = aq[:, 2] >= threshold
    high_left = True  # convencion del proyecto: meseta alta a la izquierda (col chica)
    xs_at_q = xs[aq[:, 0].astype(int)]
    signed = aq[:, 1] - xs_at_q  # >0 hacia la derecha del cruce
    dist_high = np.where(is_high, np.abs(signed), np.nan)
    dist_low = np.where(~is_high, np.abs(signed), np.nan)
    high_fit = fit_relaxation(dist_high[is_high], aq[is_high, 2])
    low_fit = fit_relaxation(dist_low[~is_high], aq[~is_high, 2])
    print("Relajacion alta  C=%.4f A=%.4f L=%.2f" % high_fit)
    print("Relajacion baja  C=%.4f A=%.4f L=%.2f" % low_fit)

    # --- mascara real: las consultas de la busqueda binaria + N_EXTRA -----
    real_mask = np.zeros((H, W), dtype=bool)
    real_mask[aq[:, 0].astype(int), aq[:, 1].astype(int)] = True

    # ANTES: dispersion gaussiana al azar alrededor de la curva -- dejaba
    # huecos de varias filas sin NINGUN punto real (la busqueda binaria solo
    # cubre ~20 filas sonda), y ahi es donde DIP alucinaba el escalonado sin
    # poder anclarlo. AHORA: rellenar sistematicamente cada fila que todavia
    # no tiene punto real, con uno pegado a la curva ya encontrada (xs[row])
    # -- cobertura real CONTINUA fila a fila, no solo en las filas sonda.
    # Pase 1: una por fila sin cobertura, pegada a la curva (cierra los
    # huecos de la busqueda binaria). Si sobra presupuesto de N_EXTRA,
    # pases siguientes agregan MAS puntos por fila (con jitter chico,
    # EXTRA_SPREAD px) para acercarse a la densidad real que usa
    # frontier_mix (328 pts) sin gastar nada en la meseta -- todo sigue
    # concentrado pegado a la frontera encontrada.
    rng = np.random.default_rng(SEED)
    added = 0
    passno = 0
    while added < N_EXTRA:
        rows_order = list(range(H))
        rng.shuffle(rows_order)
        placed_this_pass = 0
        for r in rows_order:
            if added >= N_EXTRA:
                break
            jitter = 0 if passno == 0 else rng.normal(0, EXTRA_SPREAD)
            c = int(np.clip(round(xs[r] + jitter), 0, W - 1))
            if real_mask[r, c]:
                continue
            real_mask[r, c] = True
            added += 1
            placed_this_pass += 1
        passno += 1
        if placed_this_pass == 0:  # ya no entra ningun punto nuevo (mascara llena)
            break
    n_extra = added
    n_real = int(real_mask.sum())

    # --- sinteticos: GRILLA regular en la meseta (no azar) -- la familia
    # `grid` ya viene siendo sorprendentemente competitiva toda la sesion
    # (reparto parejo, sin huecos ni amontonamientos); se aplica la misma
    # idea a los puntos INVENTADOS en vez de sortearlos al azar.
    step = max(1, int(round(np.sqrt(H * W / max(N_SYNTH, 1)))))
    off = step // 2
    grid = np.zeros((H, W), dtype=bool)
    grid[off::step, off::step] = True
    dist_to_curve = np.abs(np.arange(W)[None, :] - xs[:, None])
    candidate = grid & (dist_to_curve > MARGIN) & (~real_mask)
    cand_rows, cand_cols = np.where(candidate)
    if len(cand_rows) > N_SYNTH:
        pick = rng.choice(len(cand_rows), size=N_SYNTH, replace=False)
        cand_rows, cand_cols = cand_rows[pick], cand_cols[pick]
    n_synth = len(cand_rows)
    signed_c = cand_cols - xs[cand_rows]
    is_high_side = signed_c <= 0
    invented = np.where(
        is_high_side,
        np.clip(_relax_func(np.abs(signed_c), *high_fit), 0, 1),
        np.clip(_relax_func(np.abs(signed_c), *low_fit), 0, 1),
    )

    composite = gt.copy()
    composite[cand_rows, cand_cols] = invented
    augmented_mask = real_mask.copy()
    augmented_mask[cand_rows, cand_cols] = True

    out_dir = os.path.join(MASKS_DIR, "g%s" % g, "active_search")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(composite, 0, 1) * 255).astype(np.uint8), mode="L").save(
        os.path.join(out_dir, "image_composite.png"))
    np.save(os.path.join(out_dir, "mask_augmented.npy"), augmented_mask)
    np.save(os.path.join(out_dir, "mask_real_only.npy"), real_mask)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ry, rx = np.where(real_mask)
    axs[0].scatter(rx, ry, s=5, c="red")
    axs[0].plot(xs, np.arange(H), color="lime", linewidth=1.0)
    axs[0].set_title("%d reales (busqueda binaria + %d extra)" % (n_real, n_extra))
    axs[1].imshow(composite, cmap="gray", vmin=0, vmax=1)
    axs[1].scatter(cand_cols, cand_rows, s=4, c="cyan")
    axs[1].set_title("%d sinteticos (cian)" % n_synth)
    axs[2].imshow(np.abs(composite - gt), cmap="magma")
    axs[2].set_title("|inventado - real| (diagnostico)")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("g=%s: busqueda binaria activa -- %d reales (%d busqueda + %d extra) + %d sinteticos" % (
        g, n_real, n_binary, n_extra, n_synth))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=130)
    plt.close(fig)

    err = np.abs(composite - gt)[cand_rows, cand_cols]
    print("Reales: %d (busqueda=%d + extra=%d)   Sinteticos: %d   Total mascara: %d" % (
        n_real, n_binary, n_extra, n_synth, int(augmented_mask.sum())))
    print("Error del valor inventado vs. el real (diagnostico, MAE): %.4f  max=%.4f" % (
        err.mean(), err.max()))
    print("Escrito en", out_dir)


if __name__ == "__main__":
    main()
