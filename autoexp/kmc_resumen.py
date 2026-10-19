"""Resumen de un g terminado: todos los puntos simulados (SoC, tiempo simulado, horas de computo),
el mapa reconstruido y los puntos de control contra lo que predijo el mapa.

Uso:
    python -m autoexp.kmc_resumen --state autoexp/kmc/g-4/kmc.json [--scratch ~/.cache/kmc_auto]
-> resumen.csv (una fila por punto y tanda) y resumen.png (4 paneles) al lado de kmc.json.
Lee las corridas de <scratch>/<tag>_tXX/, <tag>_largos2/ y <tag>_control/ (lo que trae kmc_auto).
"""
import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from autoexp import kmc_planner, kmc_results

RUN = re.compile(r"(?:g[-\d.]+_)?r(\d+)_c(\d+)_k(\d+)$")


def read_runs(folder):
    """{(row, col): [dict(rc, seg, soc, tsim), ...]} de las corridas de una tanda."""
    out = {}
    for d in sorted(glob.glob(os.path.join(folder, "runs", "*"))):
        m = RUN.search(os.path.basename(d))
        est = os.path.join(d, "estado.txt")
        if not m or not os.path.exists(est):
            continue
        kv = dict(x.split("=", 1) for x in open(est).read().split() if "=" in x)
        datos = sorted(glob.glob(os.path.join(d, "datos-*.dat")))
        filas = kmc_results.read_datos(datos[0]) if datos else []  # mismo criterio que kmc_results
        soc, tsim = (filas[-1][0], filas[-1][2]) if filas else (np.nan, np.nan)
        if kv.get("rc") == "0" and not filas:
            soc, tsim = 0.0, 0.0  # termino sin filas: SoC 0 (cae al corte en el primer intervalo)
        out.setdefault((int(m.group(1)), int(m.group(2))), []).append(
            {"rc": kv.get("rc"), "seg": float(kv.get("segundos", "nan")), "soc": soc, "tsim": tsim})
    return out


def label(ax, xs, ys, texts, min_dy=0.22, **kw):
    """Etiquetas sin encimarse: en cada columna salta las que caen a menos de min_dy de la anterior."""
    done = []
    for x, y, t in sorted(zip(xs, ys, texts), key=lambda z: -z[1]):
        if any(abs(x - a) < 0.3 and abs(y - b) < min_dy for a, b in done):
            continue
        done.append((x, y))
        ax.annotate(t, (x, y), xytext=(0, 8), textcoords="offset points", ha="center", **kw)


def summarize(runs):
    ok = [r for r in runs if r["rc"] == "0"]
    use = ok or runs
    socs = np.array([r["soc"] for r in use])
    return {"soc": np.nanmean(socs) if not ok else float(np.mean(socs)),
            "std": float(np.nanstd(socs)) if len(use) > 1 else 0.0,
            "n_ok": len(ok), "n": len(runs),
            "estado": "terminado" if ok else "cortado (cota)",
            "tsim": float(np.nanmean([r["tsim"] for r in use])),
            "horas": max([r["seg"] for r in use if np.isfinite(r["seg"])] or [np.nan]) / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/kmc_auto"))
    a = ap.parse_args()
    st = kmc_planner.load(a.state)
    here = os.path.dirname(a.state)
    tag = st.get("auto_args", {}).get("tag", "g%g" % st.get("g", -4))
    g = st.get("g", -4)
    lx, le = kmc_planner.axes(st)
    ext = [le[0], le[-1], lx[-1], lx[0]]
    final = np.load(os.path.join(here, "dip", "final.npy"))
    tps = np.load(os.path.join(here, "mapa_kmc.npy"))
    pseudo = set(map(tuple, st.get("pseudo", [])))

    rows, sim = [], {}
    tandas = sorted(glob.glob(os.path.join(a.scratch, "%s_t[0-9]*" % tag))) + \
        sorted(glob.glob(os.path.join(a.scratch, "%s_largos*" % tag)))
    for folder in tandas:
        name = os.path.basename(folder)[len(tag) + 1:]
        for (i, j), runs in read_runs(folder).items():
            s = summarize(runs)
            usado = st["known_d"].get((i, j))
            rows.append({"tanda": name, "row": i, "col": j, "logxi": round(lx[i], 4), "logell": round(le[j], 4),
                         "soc_kmc": round(s["soc"], 4), "std": round(s["std"], 4),
                         "replicas_ok": s["n_ok"], "replicas": s["n"], "estado": s["estado"],
                         "tiempo_simulado": s["tsim"], "horas_computo": round(s["horas"], 3),
                         "valor_usado": "" if usado is None else round(usado, 4),
                         "tipo": "pseudo" if (i, j) in pseudo else "medido" if usado is not None else "no usado",
                         "soc_mapa_final": round(float(final[i, j]), 4)})
            if s["n_ok"] or (i, j) not in sim:
                sim[(i, j)] = s
    ctrl = read_runs(os.path.join(a.scratch, "%s_control" % tag))
    cpts = []
    for (i, j), runs in sorted(ctrl.items()):
        s = summarize(runs)
        cpts.append((i, j, s))
        rows.append({"tanda": "control", "row": i, "col": j, "logxi": round(lx[i], 4), "logell": round(le[j], 4),
                     "soc_kmc": round(s["soc"], 4), "std": round(s["std"], 4), "replicas_ok": s["n_ok"],
                     "replicas": s["n"], "estado": s["estado"], "tiempo_simulado": s["tsim"],
                     "horas_computo": round(s["horas"], 3), "valor_usado": "", "tipo": "control",
                     "soc_mapa_final": round(float(final[i, j]), 4)})
    with open(os.path.join(here, "resumen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- figura ---
    known = st["known_d"]
    meas = [p for p in known if p not in pseudo]
    ps = [p for p in known if p in pseudo]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
    kw = dict(cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")

    ax = axs[0, 0]
    im = ax.imshow(final, **kw)
    for pts, mk in ((meas, "o"), (ps, "^")):
        p = np.array(pts)
        ax.scatter(le[p[:, 1]], lx[p[:, 0]], c=[known[t] for t in pts], cmap="viridis", vmin=0, vmax=1,
                   marker=mk, s=60, edgecolors="w", linewidths=1.1)
    label(ax, [le[j] for i, j in meas + ps], [lx[i] for i, j in meas + ps],
          ["%.2f" % known[t] for t in meas + ps], fontsize=7, color="w")
    ax.plot([], [], "o", c="gray", label="KMC medido (%d)" % len(meas))
    ax.plot([], [], "^", c="gray", label="pseudo: cota + monotonía (%d)" % len(ps))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title("Mapa reconstruido (TPS + DIP) con %d puntos, g=%g" % (len(known), g))
    fig.colorbar(im, ax=ax, label="SoC")

    pts = [p for p in sim if p in known or p in pseudo]
    pts_t = [p for p in pts if sim[p]["tsim"] > 0]
    for ax, key, lab, fmt in ((axs[0, 1], "tsim", "tiempo simulado (KMC)", "%.2g"),
                              (axs[1, 0], "horas", "horas de cómputo por corrida", "%.2g")):
        ax.imshow(final, cmap="gray", vmin=0, vmax=1, extent=ext, aspect="auto", alpha=0.35)
        sel = pts_t if key == "tsim" else [p for p in pts if sim[p]["horas"] > 0]
        v = np.array([sim[p][key] for p in sel])
        p = np.array(sel)
        cut = np.array([sim[q]["n_ok"] == 0 for q in sel])
        sc = ax.scatter(le[p[:, 1]], lx[p[:, 0]], c=v, cmap="magma", norm=LogNorm(v.min(), v.max()), s=70,
                        edgecolors=np.where(cut, "r", "k"), linewidths=np.where(cut, 2, 0.6))
        label(ax, [le[q[1]] for q in sel], [lx[q[0]] for q in sel], [fmt % x for x in v], fontsize=7)
        zero = [q for q in pts if q not in sel]
        if zero:
            z = np.array(zero)
            ax.scatter(le[z[:, 1]], lx[z[:, 0]], marker="x", c="0.3", s=30, label="SoC 0 al instante")
        ax.scatter([], [], facecolors="none", edgecolors="r", linewidths=2, label="cortado por tope (cota)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        ax.set_title(lab.capitalize())
        fig.colorbar(sc, ax=ax, label=lab)

    ax = axs[1, 1]
    kmc = np.array([s["soc"] for _, _, s in cpts])
    err = np.array([s["std"] for _, _, s in cpts])
    pf = np.array([final[i, j] for i, j, _ in cpts])
    pt = np.array([tps[i, j] for i, j, _ in cpts])
    ax.plot([0, 1], [0, 1], c="0.6", lw=1)
    ax.errorbar(kmc, pf, xerr=err, fmt="o", c="C0", ms=7, label="TPS + DIP (final)")
    ax.scatter(kmc, pt, marker="x", c="C1", s=40, label="TPS sola")
    for k, (i, j, _) in enumerate(cpts):
        ax.annotate("%d" % (k + 1), (kmc[k], pf[k]), xytext=(6, -3), textcoords="offset points", fontsize=8)
    rf, rt = np.sqrt(np.mean((pf - kmc) ** 2)), np.sqrt(np.mean((pt - kmc) ** 2))
    ax.set_title("%d puntos de control (no usados): RMS %.3f (máx %.3f); TPS sola %.3f" % (
        len(cpts), rf, np.max(np.abs(pf - kmc)), rt))
    ax.set_xlabel("SoC simulado con KMC")
    ax.set_ylabel("SoC predicho por el mapa")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    c = np.array([(i, j) for i, j, _ in cpts])
    axs[0, 0].scatter(le[c[:, 1]], lx[c[:, 0]], marker="s", s=70, facecolors="none", edgecolors="r",
                      linewidths=1.6, label="control (%d)" % len(cpts))
    for k, (i, j) in enumerate(c):
        axs[0, 0].annotate("%d" % (k + 1), (le[j], lx[i]), xytext=(7, -10), textcoords="offset points",
                           fontsize=8, color="r")
    axs[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.85)
    for ax in axs.flat[:3]:
        ax.set_xlabel(r"$\log(\ell)$")
        ax.set_ylabel(r"$\log(\Xi)$")
    out = os.path.join(here, "resumen.png")
    fig.savefig(out, dpi=140)
    print("->", out, "y", os.path.join(here, "resumen.csv"))


if __name__ == "__main__":
    main()
