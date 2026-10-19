"""Mapa KMC denso de un g (p. ej. 64x64 de g=-4), sin planificador: todos los puntos de la grilla,
para ver el diagrama "en crudo" y tener verdad densa contra la cual medir las reconstrucciones.

Uso:
    python -m autoexp.kmc_grid tandas --g -4 --res 64 --parts 4 --min-logell -3.0 --out autoexp/kmc/g-4/grid64
        -> parte_1.csv ... parte_N.csv (row,col,logxi,logell,xi,ell,g): una por nodo de Serafin
    # en Serafin, cada parte en su carpeta: sbatch --time=2-00:00:00 ~/kmc/kmc_tanda.slurm tanda.csv
    # (1 replica por punto: NREP = 64 / puntos -> 1). Reenviar la misma carpeta retoma lo que falte.
    python -m autoexp.kmc_results <carpeta parte_k> --out resultados_k.csv       # por parte
    python -m autoexp.kmc_grid mapa --g -4 --res 64 --out autoexp/kmc/g-4/grid64 resultados_*.csv
        -> mapa.npy (NaN = no simulado) y mapa.png: KMC crudo | continuo | KMC - continuo

Grilla: las mismas puntas que el planificador (filas log Xi 2 -> -4, columnas log l -4 -> 2).
Las columnas con log l < --min-logell no se corren (1.5-3 dias por corrida, no entran en 48 h) y
quedan en gris. Los puntos se reparten por columna en orden de costo (lo lento primero, para que
no quede una corrida de 8 h sola al final) y en round-robin entre partes (carga pareja).
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from autoexp.oracle import load_truth


def axes(res):
    return np.linspace(2.0, -4.0, res), np.linspace(-4.0, 2.0, res)


def tandas(a):
    lx, le = axes(a.res)
    cols = [j for j in range(a.res) if le[j] >= a.min_logell]
    pts = [(i, j) for j in cols for i in range(a.res)]  # columnas de log l bajo (lentas) primero
    os.makedirs(a.out, exist_ok=True)
    parts = [pts[k::a.parts] for k in range(a.parts)]
    for k, sub in enumerate(parts, 1):
        path = os.path.join(a.out, "parte_%d.csv" % k)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "col", "logxi", "logell", "xi", "ell", "g"])
            for i, j in sub:
                w.writerow([i, j, "%.6f" % lx[i], "%.6f" % le[j], "%.6g" % 10 ** lx[i], "%.6g" % 10 ** le[j],
                            "%g" % a.g])
        print("%s: %d puntos" % (path, len(sub)))
    print("total %d puntos (%d columnas con log l >= %.2f de %d)" % (len(pts), len(cols), a.min_logell, a.res))


def continuum(g, res):
    """El continuo (128x128) llevado a la grilla res x res por interpolacion."""
    t = load_truth("%.1f" % g)
    lx128, le128 = axes(t.shape[0])
    f = RegularGridInterpolator((lx128[::-1], le128), t[::-1])
    lx, le = axes(res)
    X, L = np.meshgrid(lx, le, indexing="ij")
    return f(np.c_[X.ravel(), L.ravel()]).reshape(res, res)


def mapa(a):
    lx, le = axes(a.res)
    m = np.full((a.res, a.res), np.nan)
    cota = np.zeros((a.res, a.res), bool)
    for path in a.resultados:
        for r in csv.DictReader(open(path)):
            if r["status"] not in ("terminado", "cota"):
                continue
            i, j = int(r["row"]), int(r["col"])
            m[i, j] = float(r["value"])
            cota[i, j] = r["status"] == "cota"
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "mapa.npy"), m)
    np.save(os.path.join(a.out, "cota.npy"), cota)
    c = continuum(a.g, a.res)
    np.save(os.path.join(a.out, "continuo.npy"), c)

    ext = [le[0], le[-1], lx[-1], lx[0]]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("0.75")
    kw = dict(extent=ext, aspect="auto", interpolation="nearest")
    im = axs[0].imshow(np.ma.masked_invalid(m), cmap=cm, vmin=0, vmax=1, **kw)
    if cota.any():
        ii, jj = np.nonzero(cota)
        axs[0].scatter(le[jj], lx[ii], marker="^", s=8, c="w", lw=0, label="cota (cortada)")
        axs[0].legend(loc="lower right", fontsize=8)
    n = np.isfinite(m).sum()
    axs[0].set_title("KMC crudo %dx%d, g = %g  (%d de %d puntos)" % (a.res, a.res, a.g, n, m.size))
    fig.colorbar(im, ax=axs[0], label="SoC max")
    im = axs[1].imshow(c, cmap="viridis", vmin=0, vmax=1, **kw)
    axs[1].set_title("Continuo g = %g" % a.g)
    fig.colorbar(im, ax=axs[1], label="SoC max")
    d = np.ma.masked_invalid(np.where(cota, np.nan, m - c))  # las cotas no son el valor: fuera de la resta
    cmd = plt.get_cmap("RdBu_r").copy()
    cmd.set_bad("0.75")
    im = axs[2].imshow(d, cmap=cmd, vmin=-0.5, vmax=0.5, **kw)
    nt = n - int(cota.sum())
    rms = float(np.sqrt(np.nanmean(d.filled(np.nan) ** 2))) if nt else float("nan")
    axs[2].set_title("KMC - continuo, solo terminados  (RMS %.3f, %d puntos)" % (rms, nt))
    fig.colorbar(im, ax=axs[2], label="diferencia de SoC")
    for ax in axs:
        ax.set_xlabel("log l")
        ax.set_ylabel("log Xi")
    out = os.path.join(a.out, "mapa.png")
    fig.savefig(out, dpi=150)
    print("%d de %d puntos con valor (%d cotas); RMS vs continuo %.3f -> %s" % (n, m.size, cota.sum(), rms, out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tandas", "mapa"):
        p = sub.add_parser(name)
        p.add_argument("--g", type=float, default=-4.0)
        p.add_argument("--res", type=int, default=64)
        p.add_argument("--out", required=True)
    sub.choices["tandas"].add_argument("--parts", type=int, default=4)
    sub.choices["tandas"].add_argument("--min-logell", type=float, default=-3.0)
    sub.choices["mapa"].add_argument("resultados", nargs="+")
    a = ap.parse_args()
    {"tandas": tandas, "mapa": mapa}[a.cmd](a)


if __name__ == "__main__":
    main()
