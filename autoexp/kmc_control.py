"""Tanda de control de la reconstruccion KMC: puntos al azar que NO se usaron para reconstruir;
se corren con KMC y se comparan con el mapa (la unica medida de error real con KMC, que no
tiene mapa completo contra el cual calcular PSNR).

    python -m autoexp.kmc_control pick --state kmc.json --out control.csv [--n 10] [--min-logell -3]
    python -m autoexp.kmc_control eval --state kmc.json --results resultados_control.csv
                                       --final dip/final.npy [--out control]  -> control.png / control.json

Los puntos se sortean (semilla fija) 70 % donde la reconstruccion da SoC > 0.05 (en la zona 0
acertar es trivial) y 30 % uniformes en todo el mapa, lejos (>= 6 px) de los ya medidos, y
con log l >= --min-logell (def -2.2, <~4 h por corrida): mas abajo no termina en el tope de 8 h.
"""
import argparse
import csv
import json

import numpy as np

from autoexp import interp, kmc_planner


def cmd_pick(a):
    st = kmc_planner.load(a.state)
    lx, le = kmc_planner.axes(st)
    known = np.array(list(st["known_d"].keys()))
    rec, _ = kmc_planner.reconstruct(st)
    n_int = int(round(a.n * 0.7))  # 70 % en la zona no trivial (SoC reconstruido > 0.05), el resto uniforme
    rng = np.random.default_rng(a.seed)
    pts = []
    while len(pts) < a.n:
        i, j = rng.integers(0, st["res"], 2)
        if le[j] < a.min_logell or (len(pts) < n_int and rec[i, j] <= 0.05):
            continue
        if np.min(np.hypot(*(known - [i, j]).T)) < 6 or any(np.hypot(i - p, j - q) < 6 for p, q in pts):
            continue
        pts.append((int(i), int(j)))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        gcol = ["%g" % st["g"]] if "g" in st else []
        w.writerow(["row", "col", "logxi", "logell", "xi", "ell"] + (["g"] if gcol else []))
        for i, j in pts:
            w.writerow([i, j, "%.6f" % lx[i], "%.6f" % le[j], "%.6g" % 10 ** lx[i], "%.6g" % 10 ** le[j]] + gcol)
    print("%d puntos de control -> %s" % (len(pts), a.out))


def cmd_eval(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = kmc_planner.load(a.state)
    final = np.load(a.final)
    ij = np.array(list(st["known_d"].keys()), int)
    tps, _ = interp.cliff(ij, np.array(list(st["known_d"].values())), final.shape, sigma=st.get("sigma", 0.0))
    rows = [r for r in csv.DictReader(open(a.results)) if r["status"] == "terminado"]
    c = np.array([[int(r["row"]), int(r["col"])] for r in rows])
    kmc = np.array([float(r["value"]) for r in rows])
    out = {"n": len(rows), "sigma_kmc": st.get("sigma")}
    for name, m in (("fusion", final), ("tps", tps)):
        e = m[c[:, 0], c[:, 1]] - kmc
        rms = float(np.sqrt(np.mean(e ** 2)))
        out[name] = {"rms": rms, "mae": float(np.mean(np.abs(e))), "max": float(np.max(np.abs(e))),
                     "psnr_equiv": float(20 * np.log10(1.0 / rms)) if rms > 0 else None}
    json.dump(out, open(a.out + ".json", "w"), indent=1)

    lx, le = kmc_planner.axes(st)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    im = axs[0].imshow(final, cmap="viridis", vmin=0, vmax=1, extent=[le[0], le[-1], lx[-1], lx[0]], aspect="auto")
    axs[0].scatter(le[ij[:, 1]], lx[ij[:, 0]], s=10, c="w", alpha=0.6, label="usados (%d)" % len(ij))
    axs[0].scatter(le[c[:, 1]], lx[c[:, 0]], s=60, c=kmc, cmap="viridis", vmin=0, vmax=1, marker="s",
                   edgecolors="r", linewidths=1.5, label="control (%d)" % len(c))
    axs[0].legend(loc="lower left", fontsize=8)
    axs[0].set_xlabel(r"$\log(\ell)$")
    axs[0].set_ylabel(r"$\log(\Xi)$")
    axs[0].set_title("Mapa final y puntos de control")
    fig.colorbar(im, ax=axs[0], label="SoC")
    pred = final[c[:, 0], c[:, 1]]
    axs[1].plot([0, 1], [0, 1], c="0.6", lw=1)
    axs[1].scatter(kmc, pred, s=40, c="C0")
    axs[1].set_xlabel("SoC KMC (control)")
    axs[1].set_ylabel("SoC reconstruido")
    axs[1].set_title("Error en control: RMS %.3f, máx %.3f (TPS sola: RMS %.3f)" % (
        out["fusion"]["rms"], out["fusion"]["max"], out["tps"]["rms"]))
    axs[1].set_aspect("equal")
    fig.savefig(a.out + ".png", dpi=140)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pick")
    p.add_argument("--state", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--min-logell", type=float, default=-2.2)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("eval")
    p.add_argument("--state", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--final", required=True)
    p.add_argument("--out", default="control")
    a = ap.parse_args()
    {"pick": cmd_pick, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
