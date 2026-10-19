#!/usr/bin/env python
# coding: utf-8
"""Figuras didacticas de explicacion_autoexp (se calculan con el codigo real de autoexp/).

Todas sobre un mismo caso (g = -2.0, 64 puntos, la receta del planificador de KMC) para que se
pueda seguir el recorrido de una figura a la otra. Las usa make_explicacion_autoexp.py.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from skimage.metrics import peak_signal_noise_ratio

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from autoexp import interp, sampling  # noqa: E402
from autoexp.kmc_planner import SAMPLER_KW  # noqa: E402
from autoexp.oracle import Oracle, load_truth  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
G = "-2.0"
EXT = [-4, 2, -4, 2]  # log l (columnas), log Xi (filas, arriba = 2)

# paleta categorica validada (la misma de presentacion4_autoexp)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STAGE_C = {"grilla": C1, "bisección": C2, "relleno": C3}

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _psnr(truth, rec):
    return peak_signal_noise_ratio(truth, np.clip(rec, 0, 1), data_range=1.0)


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)


# ------------------------------------------------------- muestreo por etapas ---
class StageOracle(Oracle):
    """Oraculo que anota en que etapa del muestreo se pidio cada punto."""

    stage = "grilla"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.order = []  # (i, j, etapa, ronda)
        self.round = 0

    def query(self, points):
        before = set(self.queried)
        out = super().query(points)
        self.round += 1
        for p in dict.fromkeys((int(i), int(j)) for i, j in points):
            if p not in before:
                self.order.append((p[0], p[1], self.stage, self.round))
        return out


def run_sampling(g=G):
    orc = StageOracle(g, budget=64)
    grid0, loo0 = sampling.grid, sampling.loo_fill

    def grid_w(o, *a, **k):
        o.stage = "grilla"
        grid0(o, *a, **k)
        o.stage = "bisección"

    def loo_w(o, *a, **k):
        o.stage = "relleno"
        loo0(o, *a, **k)

    sampling.grid, sampling.loo_fill = grid_w, loo_w
    try:
        sampling.sample(orc, "bisect", 64, **SAMPLER_KW)
    finally:
        sampling.grid, sampling.loo_fill = grid0, loo0
    return orc


def fig_problema():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, g in zip(axs, ["-4.0", "-2.0", "-0.5"]):
        im = ax.imshow(load_truth(g), cmap="viridis", vmin=0, vmax=1, extent=EXT, aspect="auto")
        ax.set_title("g = %s" % g, fontsize=13)
        ax.set_xlabel(r"$\log(\ell)$")
    axs[0].set_ylabel(r"$\log(\Xi)$")
    ax = axs[1]
    ax.annotate("", xy=(-3.3, -3.5), xytext=(-3.3, 1.2),
                arrowprops=dict(arrowstyle="-|>", color="w", lw=2.2))
    ax.text(-3.1, -0.9, "SoC baja\n(o igual)", color="w", fontsize=10.5, va="center")
    ax.annotate("", xy=(1.6, 1.5), xytext=(-2.6, 1.5), arrowprops=dict(arrowstyle="-|>", color="w", lw=2.2))
    ax.text(-0.5, 1.05, "SoC baja (o igual)", color="w", fontsize=10.5, ha="center", va="top")
    axs[0].text(-1.2, -3.3, "SoC = 0 exacto\n(acantilado)", color="w", fontsize=10.5)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cb.set_label(r"SoC$_{max}$")
    return _save(fig, "problema.png")


def fig_baseline():
    truth = load_truth(G)
    orc = Oracle(G, budget=64)
    sampling.grid(orc, 64, nx=8)
    ij, v = orc.observed()
    rec = interp.reconstruct(ij, v, truth.shape, "rbf_tps")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(truth, cmap="viridis", vmin=0, vmax=1)
    axs[0].scatter(ij[:, 1], ij[:, 0], s=22, c="w", edgecolors="k", linewidths=0.5)
    axs[0].set_title("Grilla 8×8 = 64 puntos", fontsize=13)
    axs[1].imshow(rec, cmap="viridis", vmin=0, vmax=1)
    axs[1].set_title("TPS (scipy): %.1f dB" % _psnr(truth, rec), fontsize=13)
    axs[2].imshow(np.abs(rec - truth), cmap="magma", vmin=0, vmax=0.5)
    axs[2].set_title("|error|: todo en el borde", fontsize=13)
    for a in axs:
        _clean(a)
    return _save(fig, "baseline.png")


def fig_perfil():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for g, c in zip(["-4.0", "-2.0", "-0.5"], [C1, C2, C3]):
        ax.plot(load_truth(g)[:, 20], color=c, lw=2.4, label="g = %s" % g)
    ax.set_xlabel("fila (columna 20, de arriba hacia abajo = Ξ decreciente)")
    ax.set_ylabel("SoC")
    ax.annotate("meseta", (10, 1.03), fontsize=12, color=INK2)
    ax.annotate("rampa suave", (50, 0.72), fontsize=12, color=INK2)
    ax.annotate("salto a 0\n(acantilado)", (100, 0.45), fontsize=12, color=INK2)
    ax.set_ylim(-0.05, 1.12)
    ax.grid(color=GRID)
    ax.legend(loc="lower left")
    return _save(fig, "perfil.png")


def fig_muestreo(orc):
    truth = load_truth(G)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
    titles = ["1 · Grilla gruesa", "2 · + Bisección del acantilado", "3 · + Relleno por validación cruzada"]
    upto = [("grilla",), ("grilla", "bisección"), ("grilla", "bisección", "relleno")]
    for ax, t, st in zip(axs, titles, upto):
        ax.imshow(truth, cmap="gray", vmin=-0.6, vmax=1.4)
        for s in st:
            pts = np.array([(i, j) for i, j, e, _ in orc.order if e == s]).reshape(-1, 2)
            ax.scatter(pts[:, 1], pts[:, 0], s=40, c=STAGE_C[s], edgecolors="w", linewidths=0.8,
                       label="%s (%d)" % (s, len(pts)), zorder=3)
        n = sum(1 for o in orc.order if o[2] in st)
        ax.set_title("%s\n%d puntos" % (t, n), fontsize=12.5)
        _clean(ax)
    h, lab = axs[2].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.02))
    return _save(fig, "muestreo_etapas.png")


def fig_biseccion(orc):
    truth = load_truth(G)
    bis = [(i, j, r) for i, j, e, r in orc.order if e == "bisección"]
    cols = {}
    for i, j, r in bis:
        cols.setdefault(j, []).append((i, r))
    col = max(cols, key=lambda c: len(cols[c]))
    gridp = sorted(i for i, j, e, _ in orc.order if e == "grilla" and j == col)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(truth[:, col], color=INK2, lw=1.6, label="perfil real (el método no lo ve)")
    ax.scatter(gridp, truth[gridp, col], s=70, c=C1, edgecolors="w", zorder=3, label="grilla")
    seq = sorted(cols[col], key=lambda x: x[1])
    for k, (i, _) in enumerate(seq, 1):
        ax.scatter([i], [truth[i, col]], s=160, c=C2, edgecolors="w", zorder=4,
                   label="bisección" if k == 1 else None)
        ax.text(i, truth[i, col], str(k), color="w", ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=5)
    lo = max(g for g in gridp if truth[g, col] >= 0.004)
    hi = min(g for g in gridp if truth[g, col] < 0.004)
    ax.axvspan(lo, hi, color=C2, alpha=0.08)
    ax.text((lo + hi) / 2, 1.02, "corchete inicial\n(%d px)" % (hi - lo), ha="center", fontsize=10.5, color=C2)
    ax.set_xlabel("fila (columna %d)" % col)
    ax.set_ylabel("SoC")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(color=GRID)
    ax.legend(loc="lower left", fontsize=10.5)
    return _save(fig, "biseccion.png"), col, len(seq)


def fig_borde(orc):
    from scipy.spatial import Delaunay

    truth = load_truth(G)
    ij, v = orc.observed()
    _, info = interp.cliff(ij, v, truth.shape)
    coef, mids = info
    tri = Delaunay(ij.astype(float))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(truth, cmap="gray", vmin=-0.6, vmax=1.4)
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            ax.plot(ij[[s[a], s[b]], 1], ij[[s[a], s[b]], 0], color="w", lw=0.4, alpha=0.35)
    z = v < 0.004
    ax.scatter(ij[~z, 1], ij[~z, 0], s=26, c=C1, edgecolors="w", linewidths=0.5, label="SoC > 0", zorder=3)
    ax.scatter(ij[z, 1], ij[z, 0], s=26, c=INK, edgecolors="w", linewidths=0.5, label="SoC = 0", zorder=3)
    ax.scatter(mids[:, 1], mids[:, 0], s=60, marker="x", c=C2, linewidths=2, label="punto de borde", zorder=4)
    xs = np.arange(128)
    ax.plot(xs, interp.edge(coef, xs), color=C2, lw=2, label="borde (RANSAC)")
    ax.set_xlim(0, 127)
    ax.set_ylim(127, 0)
    _clean(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10.5)
    return _save(fig, "borde.png")


def fig_loo(orc):
    """Primera ronda del relleno: error leave-one-out de lo que habia al terminar la biseccion
    (misma cuenta que sampling.loo_fill) y los 4 puntos que eligio."""
    from scipy.interpolate import RBFInterpolator

    truth = load_truth(G)
    H, W = truth.shape
    prev = [(i, j) for i, j, e, _ in orc.order if e != "relleno"]
    fill = [(i, j, r) for i, j, e, r in orc.order if e == "relleno"]
    first = [(i, j) for i, j, r in fill if r == fill[0][2]]
    ij = np.array(prev)
    v = np.array([orc.queried[p] for p in prev])
    kw = dict(mono=False, bounds=False, smoothing=0.0)
    coef = interp.cliff(ij, v, truth.shape, **kw)[1][0]
    line = lambda r, c: interp.edge(coef, c) - r
    up = np.where(line(ij[:, 0], ij[:, 1]) > 4.0)[0]
    errs = []
    for k in up:
        m = np.ones(len(ij), bool)
        m[k] = False
        rec, _ = interp.cliff(ij[m], v[m], truth.shape, **kw)
        errs.append(abs(rec[ij[k, 0], ij[k, 1]] - v[k]))
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.clip(RBFInterpolator(ij[up] / 128.0, np.array(errs), kernel="linear")(
        np.column_stack([yy.ravel(), xx.ravel()]) / 128.0).reshape(H, W), 0, None)
    field[line(yy, xx) <= 4.0] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(field, cmap="magma", vmin=0)
    ax.scatter(ij[:, 1], ij[:, 0], s=16, c="w", edgecolors="k", linewidths=0.4, label="ya medidos")
    f = np.array(first)
    ax.scatter(f[:, 1], f[:, 0], s=150, marker="*", c=C3, edgecolors="w", linewidths=0.8, zorder=4,
               label="elegidos (ronda 1 del relleno)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="|error| al sacar cada punto (LOO)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10)
    _clean(ax)
    return _save(fig, "loo.png")


def fig_recon(orc):
    truth = load_truth(G)
    ij, v = orc.observed()
    S = truth.shape
    steps = [
        ("TPS común", interp.reconstruct(ij, v, S, "rbf_tps")),
        ("+ cero debajo del borde", interp.cliff(ij, v, S, vert=1.0, along=1.0, adapt=False, mono=False,
                                                 bounds=False)[0]),
        ("+ y comprimida\n+ rampa alineada", interp.cliff(ij, v, S, mono=False, bounds=False)[0]),
        ("+ monotonía\n(cotas + proyección)", interp.cliff(ij, v, S)[0]),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.2))
    out = []
    for k, (t, r) in enumerate(steps):
        p = _psnr(truth, r)
        out.append((t, p))
        axs[0, k].imshow(r, cmap="viridis", vmin=0, vmax=1)
        axs[0, k].set_title("%s\n%.1f dB" % (t, p), fontsize=12.5)
        axs[1, k].imshow(np.abs(r - truth), cmap="magma", vmin=0, vmax=0.3)
        _clean(axs[0, k])
        _clean(axs[1, k])
    axs[1, 0].set_ylabel("|error|", fontsize=12)
    axs[0, 0].set_ylabel("reconstrucción", fontsize=12)
    return _save(fig, "recon_pasos.png"), out


def fig_monotonia(orc):
    truth = load_truth(G)
    ij, v = orc.observed()
    L, U = interp.monotone_bounds(ij, v, truth.shape)
    w = U - L
    frac = float(np.mean(w <= 0.05))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(w, cmap="magma_r", vmin=0, vmax=1)
    ax.scatter(ij[:, 1], ij[:, 0], s=14, c="w", edgecolors="k", linewidths=0.4)
    fig.colorbar(im, ax=ax, fraction=0.046, label="ancho del intervalo U − L")
    ax.set_title("Lo que ya se sabe SOLO por monotonía", fontsize=12.5)
    _clean(ax)
    return _save(fig, "monotonia.png"), frac


def main():
    orc = run_sampling()
    fig_problema()
    fig_baseline()
    fig_perfil()
    fig_muestreo(orc)
    _, col, nb = fig_biseccion(orc)
    fig_borde(orc)
    fig_loo(orc)
    _, pasos = fig_recon(orc)
    _, frac = fig_monotonia(orc)
    cnt = {s: sum(1 for o in orc.order if o[2] == s) for s in STAGE_C}
    info = {"etapas": cnt, "col_biseccion": col, "n_biseccion_col": nb, "pasos": pasos,
            "frac_mono": frac, "rondas": orc.round}
    print(info)
    return info


if __name__ == "__main__":
    main()
