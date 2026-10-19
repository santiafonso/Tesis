#!/usr/bin/env python
# coding: utf-8
"""Figuras de presentacion6: estado del KMC en los 13 g (-8 a 4), la grilla 64x64 de g = -4 y el
control nuevo. Lee los estados de los loops (autoexp/kmc/g*/kmc.json, copia del estado vivo de
Mendieta) y, si estan, los resultados parciales de la grilla (--grid: carpeta con resultados_*.csv).

Uso (desde ~/Tesis-autoexp):
    ./venv/bin/python presentaciones/presentacion6_05_10/figuras.py [--grid <carpeta con resultados_*.csv>]
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from autoexp import kmc_control, kmc_planner  # noqa: E402
from autoexp.oracle import load_truth  # noqa: E402

KMC = os.path.join(REPO, "autoexp", "kmc")
FIG = os.path.join(HERE, "fig")
GS = list(range(-8, 5))
C_NEG, C_POS = "#2a6fdb", "#e8743b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def gname(g):
    return "g%d" % g


def state(g):
    return kmc_planner.load(os.path.join(KMC, gname(g), "kmc.json"))


def n_meas(st):
    ps = set(map(tuple, st.get("pseudo", [])))
    return sum(1 for p in st["known_d"] if p not in ps)


def pending(st):
    return [tuple(p) for p in st.get("pending", []) if tuple(p) not in st["known_d"]]


def fig_avance():
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for k, g in enumerate(GS):
        st = state(g)
        n, p, b = n_meas(st), len(pending(st)), st["budget"]
        c = C_NEG if g < 0 else C_POS
        ax.barh(k, b, color="0.92", height=0.7)
        ax.barh(k, n, color=c, height=0.7)
        ax.barh(k, p, left=n, color=c, alpha=0.35, height=0.7, hatch="///", edgecolor="w", lw=0)
        ax.text(b + 1.5, k, "%d / %d" % (n, b) + ("   (+%d corriendo)" % p if p else ""), va="center", fontsize=10,
                color="0.25")
    ax.set_yticks(range(len(GS)), ["g = %+d" % g if g else "g = 0" for g in GS])
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xlabel("puntos de KMC medidos (de 96 por g)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Avance del KMC por g (sólido: terminados · rayado: corriendo)", loc="left", fontsize=12)
    out = os.path.join(FIG, "avance_g.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _panel(ax, img, st, pts_pend=(), title=""):
    lx, le = kmc_planner.axes(st)
    ax.imshow(np.clip(img, 0, 1), cmap="viridis", vmin=0, vmax=1, extent=[le[0], le[-1], lx[-1], lx[0]],
              aspect="auto")
    ps = set(map(tuple, st.get("pseudo", [])))
    ij = np.array([p for p in st["known_d"] if p not in ps]).reshape(-1, 2)
    if len(ij):
        ax.scatter(le[ij[:, 1]], lx[ij[:, 0]], s=9, c="w", edgecolors="k", linewidths=0.4)
    if len(pts_pend):
        pp = np.array(pts_pend)
        ax.scatter(le[pp[:, 1]], lx[pp[:, 0]], s=22, marker="x", c="r", linewidths=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([-4, -2, 0, 2])
    ax.set_yticks([-4, -2, 0, 2])
    ax.tick_params(labelsize=8)


def fig_kmc(gs, out_name, sup):
    """Arriba: reconstruccion con los puntos de KMC que hay (blancos) y los que corren (x rojas).
    Abajo: el continuo del mismo g, solo como referencia (no es la verdad del KMC)."""
    fig, axs = plt.subplots(2, len(gs), figsize=(3.0 * len(gs), 5.8), constrained_layout=True, squeeze=False)
    for k, g in enumerate(gs):
        st = state(g)
        n = n_meas(st)
        rec = kmc_planner.reconstruct(st)[0] if n >= 20 else np.full((st["res"], st["res"]), np.nan)
        _panel(axs[0, k], rec, st, pending(st), "KMC g = %+d: %d puntos" % (g, n) if g else "KMC g = 0: %d puntos" % n)
        if n < 20:
            axs[0, k].text(0.5, 0.5, "primera tanda\ncorriendo", transform=axs[0, k].transAxes, ha="center",
                           va="center", fontsize=11, color="0.3")
            axs[0, k].set_facecolor("0.93")
        _panel(axs[1, k], load_truth("%.1f" % g), {"res": 128, "known_d": {}}, title="continuo g = %g" % g)
    axs[0, 0].set_ylabel("log Ξ")
    axs[1, 0].set_ylabel("log Ξ")
    for ax in axs[1]:
        ax.set_xlabel("log ℓ")
    fig.suptitle(sup, fontsize=11, x=0.01, ha="left")
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_control(g=-4):
    """Donde caerian los 32 puntos de control de g (mismo sorteo que kmc_control pick)."""
    import types

    tmp = os.path.join(FIG, "_control_g%d.csv" % g)
    a = types.SimpleNamespace(state=os.path.join(KMC, gname(g), "kmc.json"), out=tmp, n=32, frac_crit=0.75,
                              min_dist=4, min_logell=-2.7, seed=0)
    kmc_control.cmd_pick(a)
    out = os.path.join(FIG, "control_critico_g%d.png" % g)
    os.replace(tmp.replace(".csv", "_critico.png"), out)
    for f in (tmp, tmp.replace(".csv", "_tipos.json")):
        os.remove(f)
    return out


def fig_grid(d):
    """Grilla 64x64 de g = -4 en curso: solo lo terminado tiene color; lo que corre, gris oscuro."""
    res = 64
    lx, le = np.linspace(2.0, -4.0, res), np.linspace(-4.0, 2.0, res)
    m = np.full((res, res), np.nan)
    run = np.zeros((res, res), bool)
    for path in glob.glob(os.path.join(d, "resultados_*.csv")):
        for r in csv.DictReader(open(path)):
            i, j = int(r["row"]), int(r["col"])
            if r["status"] == "terminado":
                m[i, j] = float(r["value"])
            else:
                run[i, j] = True
    plan = np.zeros((res, res), bool)
    plan[:, le >= -3.0] = True
    img = np.ma.masked_invalid(m)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad((0, 0, 0, 0))
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ext = [le[0], le[-1], lx[-1], lx[0]]
    bg = np.where(plan, 0.88, 0.97)
    bg = np.where(run, 0.45, bg)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=ext, aspect="auto", interpolation="nearest")
    im = ax.imshow(img, cmap=cm, vmin=0, vmax=1, extent=ext, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="SoC máx (KMC)")
    n, nr, nt = int(np.isfinite(m).sum()), int(run.sum()), int(plan.sum())
    ax.set_title("g = −4, grilla 64×64: %d terminados, %d corriendo, de %d" % (n, nr, nt), fontsize=11)
    ax.set_xlabel("log ℓ")
    ax.set_ylabel("log Ξ")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="0.45", label="corriendo"), Patch(color="0.88", label="en cola"),
                       Patch(color="0.97", label="no se corre (log ℓ < −3)", ec="0.7")],
              loc="lower right", fontsize=8)
    out = os.path.join(FIG, "grid64_parcial.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    json.dump({"terminados": n, "corriendo": nr, "total": nt}, open(os.path.join(FIG, "grid64_parcial.json"), "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", help="carpeta con resultados_*.csv de la grilla 64x64 (kmc_results por parte)")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    print(fig_avance())
    print(fig_kmc([-1, -2, -3, -4], "kmc_neg_1a4.png",
                  "Arriba: mapa reconstruido con los puntos de KMC que hay hoy (blancos; x rojas: corriendo). "
                  "Abajo: el continuo, solo como referencia."))
    print(fig_kmc([-5, -6, -7, -8], "kmc_neg_5a8.png",
                  "g = −5 a −8 con KMC: primera tanda (grilla de 30 puntos, x rojas) corriendo desde el 5/10."))
    print(fig_kmc([0, 1, 2, 3, 4], "kmc_pos.png",
                  "Arriba: mapa reconstruido con los puntos de KMC que hay hoy. Abajo: el continuo (referencia)."))
    print(fig_control(-4))
    if a.grid:
        print(fig_grid(a.grid))


if __name__ == "__main__":
    main()
