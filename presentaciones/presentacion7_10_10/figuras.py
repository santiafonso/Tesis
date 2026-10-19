#!/usr/bin/env python
# coding: utf-8
"""Figuras de presentacion7: avance de la reconstruccion con KMC en los 13 g (-8 a 4).
Lee los estados de los loops (autoexp/kmc/g*/kmc.json, control.json, dip/final.npy: copia del estado vivo
de Mendieta) y el mapa de la grilla 64x64 (autoexp/kmc/g-4/grid64/<--grid>/mapa.png).

Uso (desde ~/Tesis-autoexp, despues de traer el estado de Mendieta):
    ./venv/bin/python presentaciones/presentacion7_10_10/figuras.py [--grid avance_1010]
"""
import argparse
import json
import os
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from autoexp import kmc_planner  # noqa: E402
from autoexp.oracle import load_truth  # noqa: E402

KMC = os.path.join(REPO, "autoexp", "kmc")
FIG = os.path.join(HERE, "fig")
GS = list(range(-8, 5))
C_MED, C_PSE, C_CRI, C_UNI = "#2a6fdb", "#9db8e8", "#d1495b", "#edae49"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def gdir(g):
    return os.path.join(KMC, "g%d" % g)


def gtxt(g):
    return "g = %+d" % g if g else "g = 0"


def state(g):
    return kmc_planner.load(os.path.join(gdir(g), "kmc.json"))


def pseudo(st):
    return set(map(tuple, st.get("pseudo", [])))


def n_meas(st):
    ps = pseudo(st)
    return sum(1 for p in st["known_d"] if p not in ps)


def etapa(g):
    """Del auto.log del loop: 'terminado' (mapa final + control), 'control' (mapa final, control
    corriendo) o 'muestreo'. Lo ultimo que paso manda: g=-4 y g=3 tienen un control viejo (64 puntos)
    y despues volvieron a muestrear."""
    last = {"muestreo": -1, "control": -1, "terminado": -1}
    for k, line in enumerate(open(os.path.join(gdir(g), "auto.log"), errors="replace")):
        if "LISTO control" in line:
            last["terminado"] = k
        elif "_control:" in line:
            last["control"] = k
        elif re.search(r"_t\d+:", line):
            last["muestreo"] = k
    return max(last, key=last.get)


def control(g):
    p = os.path.join(gdir(g), "control.json")
    return json.load(open(p)) if etapa(g) == "terminado" and os.path.exists(p) else None


def resumen():
    rows = []
    for g in GS:
        st = state(g)
        c = control(g) or {}
        rows.append({"g": g, "medidos": n_meas(st), "pseudo": len(pseudo(st)), "etapa": etapa(g),
                     "rms": c.get("fusion", {}).get("rms"), "rms_tps": c.get("tps", {}).get("rms"),
                     "rms_crit": c.get("fusion_critico", {}).get("rms"),
                     "rms_unif": c.get("fusion_uniforme", {}).get("rms"),
                     "max": c.get("fusion", {}).get("max"),
                     "soc_max": max(v for k, v in st["known_d"].items() if k not in pseudo(st))})
    json.dump(rows, open(os.path.join(FIG, "resumen.json"), "w"), indent=1)
    return rows


def fig_estado(rows):
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    lab = {"terminado": "terminado (mapa + control)", "control": "mapa listo, control corriendo",
           "muestreo": "eligiendo puntos"}
    for k, r in enumerate(rows):
        ax.barh(k, r["medidos"], color=C_MED, height=0.7)
        ax.barh(k, r["pseudo"], left=r["medidos"], color=C_PSE, height=0.7)
        ax.text(r["medidos"] + r["pseudo"] + 2, k, "%d + %d   %s" % (r["medidos"], r["pseudo"], lab[r["etapa"]]),
                va="center", fontsize=10, color="0.2" if r["etapa"] == "terminado" else "0.45",
                fontweight="bold" if r["etapa"] == "terminado" else "normal")
    ax.axvline(96, color="0.6", ls=":", lw=1)
    ax.text(96, len(rows) - 0.3, "96", ha="center", va="top", fontsize=9, color="0.4")
    ax.set_yticks(range(len(rows)), [gtxt(r["g"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 260)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel("puntos de KMC")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=C_MED, label="medidos (corrida terminada)"),
                       Patch(color=C_PSE, label="cotas (corrida cortada por tiempo)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=10, frameon=False)
    out = os.path.join(FIG, "estado_g.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _mapa(ax, img, st=None, title="", cmap="viridis", vmin=0, vmax=1, pts=True):
    res = img.shape[0]
    lx, le = (kmc_planner.axes(st) if st else (np.linspace(2.0, -4.0, res), np.linspace(-4.0, 2.0, res)))
    im = ax.imshow(img if cmap != "viridis" else np.clip(img, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[le[0], le[-1], lx[-1], lx[0]], aspect="auto")
    if st is not None and pts:
        ps = pseudo(st)
        ij = np.array([p for p in st["known_d"] if p not in ps]).reshape(-1, 2)
        if len(ij):
            ax.scatter(le[ij[:, 1]], lx[ij[:, 0]], s=7, c="w", edgecolors="k", linewidths=0.3)
        pp = np.array(sorted(ps)).reshape(-1, 2)
        if len(pp):
            ax.scatter(le[pp[:, 1]], lx[pp[:, 0]], s=14, facecolors="none", edgecolors="r", linewidths=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([-4, -2, 0, 2])
    ax.set_yticks([-4, -2, 0, 2])
    ax.tick_params(labelsize=8)
    return im


def fig_finales(gs, out_name):
    """Arriba: mapa final de KMC (TPS + DIP) con sus puntos. Medio: el continuo (referencia).
    Abajo: KMC - continuo."""
    fig, axs = plt.subplots(3, len(gs), figsize=(2.75 * len(gs) + 0.8, 8.0), constrained_layout=True, squeeze=False)
    for k, g in enumerate(gs):
        st = state(g)
        fin = np.load(os.path.join(gdir(g), "dip", "final.npy"))
        con = load_truth("%.1f" % g)
        _mapa(axs[0, k], fin, st, "KMC %s (%d medidos)" % (gtxt(g), n_meas(st)))
        _mapa(axs[1, k], con, title="continuo %s" % gtxt(g))
        imd = _mapa(axs[2, k], fin - con, st, "KMC − continuo", cmap="RdBu_r", vmin=-0.5, vmax=0.5, pts=False)
    im0 = axs[0, -1].images[0]
    fig.colorbar(im0, ax=axs[:2, -1], label="SoC máx", shrink=0.8)
    fig.colorbar(imd, ax=axs[2, -1], label="diferencia", shrink=0.9)
    for r, lab in enumerate(["KMC", "continuo", "diferencia"]):
        axs[r, 0].set_ylabel("log Ξ")
    for ax in axs[2]:
        ax.set_xlabel("log ℓ")
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_en_curso(gs, out_name):
    """Mapas de los g que todavia eligen puntos, reconstruidos con lo que hay hoy."""
    fig, axs = plt.subplots(2, 4, figsize=(12.5, 6.2), constrained_layout=True)
    for ax in axs.flat[len(gs):]:
        ax.axis("off")
    for ax, g in zip(axs.flat, gs):
        st = state(g)
        rec = kmc_planner.reconstruct(st)[0]
        im = _mapa(ax, rec, st, "KMC %s: %d medidos" % (gtxt(g), n_meas(st)))
    for ax in axs[:, 0]:
        ax.set_ylabel("log Ξ")
    for ax in axs[1]:
        ax.set_xlabel("log ℓ")
    fig.colorbar(im, ax=axs, label="SoC máx", shrink=0.6)
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_control(rows):
    rr = [r for r in rows if r["rms"] is not None]
    x = np.arange(len(rr))
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    w = 0.38
    ax.bar(x - w / 2, [r["rms_crit"] for r in rr], w, color=C_CRI, label="24 puntos en la zona crítica")
    ax.bar(x + w / 2, [r["rms_unif"] for r in rr], w, color=C_UNI, label="8 puntos al azar")
    for i, r in enumerate(rr):
        ax.text(i - w / 2, r["rms_crit"] + 0.001, "%.3f" % r["rms_crit"], ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, r["rms_unif"] + 0.001, "%.3f" % r["rms_unif"], ha="center", va="bottom", fontsize=9)
    ax.axhline(0.005, color="0.4", ls="--", lw=1, label="ruido del KMC (σ entre réplicas)")
    ax.set_xticks(x, [gtxt(r["g"]) for r in rr])
    ax.set_ylabel("RMS en los puntos de control (SoC)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("Error de la reconstrucción en 32 puntos de KMC no usados", loc="left", fontsize=12)
    out = os.path.join(FIG, "control_rms.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_techo(rows):
    g = np.array([r["g"] for r in rows])
    m = np.array([r["soc_max"] for r in rows])
    con = np.array([load_truth("%.1f" % x).max() for x in g])
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(g, con, "o-", color="0.6", label="continuo")
    ax.plot(g, m, "o-", color=C_MED, label="KMC (máximo medido)")
    for x, y in zip(g, m):
        if x >= 1:
            ax.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9,
                        color=C_MED)
    ax.set_xlabel("g (interacción de Frumkin)")
    ax.set_ylabel("SoC máximo del mapa")
    ax.set_ylim(0.8, 1.02)
    ax.set_xticks(range(-8, 5))
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("El techo del KMC baja con g > 0", loc="left", fontsize=12)
    out = os.path.join(FIG, "techo_g.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="avance_1010", help="carpeta en autoexp/kmc/g-4/grid64/ con mapa.png")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    rows = resumen()
    print(fig_estado(rows))
    fin = [r["g"] for r in rows if r["etapa"] != "muestreo"]
    print(fig_finales([g for g in fin if g < -4], "finales_neg_5a8.png"))
    print(fig_finales([g for g in fin if g >= -4], "finales_resto.png"))
    print(fig_en_curso([r["g"] for r in rows if r["etapa"] == "muestreo"], "en_curso.png"))
    print(fig_control(rows))
    print(fig_techo(rows))
    for g in (-2, -8):
        shutil.copy(os.path.join(gdir(g), "control.png"), os.path.join(FIG, "control_g%d.png" % g))
    shutil.copy(os.path.join(KMC, "g-4", "grid64", a.grid, "mapa.png"), os.path.join(FIG, "grid64.png"))


if __name__ == "__main__":
    main()
