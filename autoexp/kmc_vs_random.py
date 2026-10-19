"""Comparativa en el continuo: el procedimiento del KMC (kmc_planner: grilla + biseccion + relleno)
contra el mismo numero de puntos sorteados al azar, con la MISMA reconstruccion
(kmc_planner.reconstruct) y el mismo ruido de KMC. Responde "¿cuanto gana elegir los puntos vs
tirarlos al azar?" para cada g.

    python -m autoexp.kmc_vs_random [--gs -8 ... 4] [--budget 96] [--sigma 0.005] [--seeds 10]
                                    [--out autoexp/kmc/comparacion_random]
-> psnr.csv, psnr.png (PSNR por g: procedimiento vs azar, media +- desvio entre semillas) y
   mapas.png (por g: continuo | procedimiento | azar de la semilla mediana, con los puntos).
Los g sin mapa del continuo (results/phase_diagram_g/g<val>/sim_128.png) se saltean con aviso.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from autoexp import interp, kmc_planner
from autoexp.oracle import Oracle, load_truth


def run_planner(g, budget, sigma):
    orc = Oracle(g, budget=budget, noise=sigma)
    st = {"res": 128, "budget": budget, "sigma": sigma, "sampler_kw": kmc_planner.SAMPLER_KW, "known_d": {},
          "rounds": []}
    while True:
        pts = kmc_planner.next_batch(st)
        if not pts:
            break
        st["known_d"].update({p: float(x) for p, x in zip(pts, orc.query(pts))})
        st["rounds"].append(len(pts))
    return st


def run_random(g, budget, sigma, seed):
    orc = Oracle(g, budget=budget, noise=sigma)
    rng = np.random.default_rng(seed)
    flat = rng.choice(128 * 128, size=budget, replace=False)
    pts = [(int(f // 128), int(f % 128)) for f in flat]
    st = {"res": 128, "budget": budget, "sigma": sigma, "sampler_kw": kmc_planner.SAMPLER_KW, "rounds": [],
          "known_d": {p: float(x) for p, x in zip(pts, orc.query(pts))}}
    return st


def psnr(truth, st):
    rec, _ = kmc_planner.reconstruct(st)
    return float(peak_signal_noise_ratio(truth, np.clip(rec, 0, 1), data_range=1.0)), rec


def psnr_tps(truth, st):
    """Referencia generica para puntos sueltos: TPS sola (sin borde, cotas ni monotonia)."""
    ij = np.array(list(st["known_d"]))
    rec = interp.reconstruct(ij, np.array(list(st["known_d"].values())), truth.shape)
    return float(peak_signal_noise_ratio(truth, np.clip(rec, 0, 1), data_range=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", type=float, nargs="+", default=list(range(-8, 5)))
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--sigma", type=float, default=0.005)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="autoexp/kmc/comparacion_random")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, maps = [], []
    for gv in a.gs:
        g = "%.1f" % gv
        try:
            truth = load_truth(g)
        except FileNotFoundError:
            print("g=%s: sin mapa del continuo, se saltea" % g)
            continue
        st_p = run_planner(g, a.budget, a.sigma)
        p_plan, rec_p = psnr(truth, st_p)
        rand, tps = [], []
        for s in range(a.seeds):
            st_r = run_random(g, a.budget, a.sigma, s)
            rand.append(psnr(truth, st_r) + (s,))
            tps.append(psnr_tps(truth, st_r))
        ps, tps = np.array([r[0] for r in rand]), np.array(tps)
        med = rand[int(np.argsort(ps)[len(ps) // 2])]
        st_med = run_random(g, a.budget, a.sigma, med[2])
        rows.append({"g": gv, "psnr_procedimiento": p_plan, "psnr_azar_media": ps.mean(), "psnr_azar_std": ps.std(),
                     "psnr_azar_min": ps.min(), "psnr_azar_max": ps.max(), "psnr_azar_tps_media": tps.mean(),
                     "ganancia_db": p_plan - max(ps.mean(), tps.mean())})
        maps.append((gv, truth, rec_p, st_p, med[1], st_med, p_plan, med[0]))
        print("g=%5s: procedimiento %.2f dB | azar %.2f +- %.2f dB, azar+TPS %.2f dB (%d semillas) | +%.2f dB" % (
            g, p_plan, ps.mean(), ps.std(), tps.mean(), a.seeds, rows[-1]["ganancia_db"]))
    if not rows:
        return
    with open(os.path.join(a.out, "psnr.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    g = np.array([r["g"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    m, s = np.array([r["psnr_azar_media"] for r in rows]), np.array([r["psnr_azar_std"] for r in rows])
    ax.fill_between(g, m - s, m + s, color="C1", alpha=0.25, lw=0)
    ax.plot(g, m, "o-", c="C1", label="%d puntos al azar (media ± desvío, %d semillas)" % (a.budget, a.seeds))
    ax.plot(g, [r["psnr_azar_tps_media"] for r in rows], "^--", c="C2", label="al azar + TPS sola (media)")
    ax.plot(g, [r["psnr_procedimiento"] for r in rows], "s-", c="C0", label="procedimiento (%d puntos)" % a.budget)
    ax.set_xlabel("g")
    ax.set_ylabel("PSNR vs continuo [dB]")
    ax.set_xticks(g)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Elegir los puntos vs tirarlos al azar (continuo, σ = %g)" % a.sigma)
    fig.savefig(os.path.join(a.out, "psnr.png"), dpi=150)

    lx, le = kmc_planner.axes({"res": 128})
    ext = [le[0], le[-1], lx[-1], lx[0]]
    kw = dict(cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")
    fig, axs = plt.subplots(len(maps), 3, figsize=(10, 3 * len(maps)), constrained_layout=True, squeeze=False)
    for row, (gv, truth, rec_p, st_p, rec_r, st_r, pp, pr) in zip(axs, maps):
        row[0].imshow(truth, **kw)
        row[0].set_title("continuo g = %g" % gv, fontsize=9)
        for ax, rec, st, name, p in ((row[1], rec_p, st_p, "procedimiento", pp), (row[2], rec_r, st_r, "azar", pr)):
            ax.imshow(np.clip(rec, 0, 1), **kw)
            ij = np.array(list(st["known_d"]))
            ax.scatter(le[ij[:, 1]], lx[ij[:, 0]], s=5, c="r")
            ax.set_title("%s: %.1f dB" % (name, p), fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(os.path.join(a.out, "mapas.png"), dpi=110)
    print("->", a.out)


if __name__ == "__main__":
    main()
