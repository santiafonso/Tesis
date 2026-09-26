"""Reconstruccion final del KMC real con TPS + DIP (fusion + guarda), la receta validada:
DIP "tesis" (pseudo-ceros debajo del acantilado + franja de 6 px con la TPS, post_zero,
8000 it, LR 1e-3, REG_NOISE_STD 0.08, SKIP_N11 4), 2 semillas y mejor de 2 por residuo en los
puntos reales, fusion por distancia al acantilado (B=20) con TPS suavizada (sigma), y guarda
0.02 + 2*sigma: si DIP no reproduce los puntos reales de arriba del acantilado, solo TPS.
Ruido de la validacion: sigma .01 -> +0.7 dB sobre la TPS, .03 -> +0.7, .05 -> +1.2.

Pasos:
    python -m autoexp.kmc_dip prepare --state autoexp/kmc/g-4/kmc.json --out autoexp/kmc/g-4/dip
        -> dip/s<seed>/gkmc/{queries.json,target.png,mask.npy,env.json}
    (Mendieta, desde ~/Tesis-autoexp con la carpeta sincronizada)
    sbatch --export=ALL,DIPDIR=autoexp/kmc/g-4/dip slurm/kmc_dip.slurm
        -> dip/s<seed>/gkmc/restored.npy
    python -m autoexp.kmc_dip fuse --state autoexp/kmc/g-4/kmc.json --out autoexp/kmc/g-4/dip
        -> dip/final.npy, dip/final.png (TPS | DIP | fusion con los puntos)
Los pseudo-puntos del planificador (cotas + monotonia) entran como puntos.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from autoexp import fuse, interp, kmc_planner

SEEDS = (42, 43)
DIP_ENV = {"NUM_ITER": 8000, "LR": 0.001, "REG_NOISE_STD": 0.08, "SKIP_N11": 4, "PSNR_DROP_TOL": -8,
           "MAX_FALLBACKS": 3, "SHOW_EVERY": 200, "SNAPSHOT_EVERY": 10 ** 9, "N_CHANNELS": 1}
N_ZERO, BAND, B = 4096, 6, 20.0


def points(st):
    ij = np.array(list(st["known_d"].keys()), int)
    v = np.array(list(st["known_d"].values()), float)
    return ij, v


def cmd_prepare(a):
    st = kmc_planner.load(a.state)
    sigma = st.get("sigma", 0.0)
    ij, v = points(st)
    H = W = st["res"]
    target, info = interp.cliff(ij, v, (H, W), sigma=sigma)
    mask = np.zeros((H, W), bool)
    mask[ij[:, 0], ij[:, 1]] = True
    below = np.zeros((H, W), bool)
    if info is not None:  # igual que trial.prepare_g con aug zero+band
        yy, xx = np.mgrid[0:H, 0:W]
        below = yy > interp.edge(info[0], xx)
        step = max(1, int(round(np.sqrt(H * W / N_ZERO))))
        sub = np.zeros((H, W), bool)
        sub[step // 2::step, step // 2::step] = True
        dist = interp.edge(info[0], xx) - yy
        mask |= (below | ((dist > 0) & (dist <= BAND))) & sub
    else:
        print("OJO: no se detecto acantilado; DIP sin pseudo-ceros")
    target[ij[:, 0], ij[:, 1]] = v
    for s in SEEDS:
        d = os.path.join(a.out, "s%d" % s, "gkmc")
        os.makedirs(d, exist_ok=True)
        json.dump({"g": "kmc", "budget": st["budget"], "noise": sigma, "shape": [H, W],
                   "points": [[int(i), int(j), float(x)] for (i, j), x in zip(ij, v)]},
                  open(os.path.join(d, "queries.json"), "w"))
        Image.fromarray(np.round(np.clip(target, 0, 1) * 255).astype(np.uint8), "L").save(os.path.join(d, "target.png"))
        np.save(os.path.join(d, "mask.npy"), mask)
        np.save(os.path.join(d, "post_zero.npy"), below)
        json.dump({k: str(x) for k, x in dict(DIP_ENV, SEED=s).items()}, open(os.path.join(d, "env.json"), "w"))
    print("%d puntos (%d pseudo), mascara %d px, acantilado %s -> %s/s{%s}" % (
        len(ij), len(st.get("pseudo", [])), mask.sum(), "si" if info is not None else "no", a.out,
        ",".join(map(str, SEEDS))))


def cmd_fuse(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = kmc_planner.load(a.state)
    sigma = st.get("sigma", 0.0)
    rk = {"sigma": sigma}
    srcs = [os.path.join(a.out, "s%d" % s) for s in SEEDS]
    for s in srcs:  # post_zero como en trial.run (la salida de DIP trae el pad de 1 px)
        g = os.path.join(s, "gkmc")
        r = np.load(os.path.join(g, "restored.npy"))
        r = r[0] if r.ndim == 3 else r
        r = r[1:-1, 1:-1] if r.shape[0] == st["res"] + 2 else r
        r[np.load(os.path.join(g, "post_zero.npy"))] = 0.0
        np.save(os.path.join(g, "restored.npy"), r)
    best = os.path.join(a.out, "mejor_de_2")
    fuse.best_of(srcs, best, recon_kw=rk)
    fin = os.path.join(a.out, "fusion")
    fuse.fuse_run(best, fin, B=B, guard=0.02 + 2 * sigma, recon_kw=rk)
    final = np.load(os.path.join(fin, "gkmc", "restored.npy"))
    np.save(os.path.join(a.out, "final.npy"), final)

    ij, v = points(st)
    tps, _ = interp.cliff(ij, v, final.shape, sigma=sigma)
    dip = np.load(os.path.join(best, "gkmc", "restored.npy"))
    lx, le = kmc_planner.axes(st)
    ext = [le[0], le[-1], lx[-1], lx[0]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, im_, t in zip(axs, (tps, dip, final), ("TPS", "DIP (mejor de 2)", "Fusión TPS + DIP (final)")):
        im = ax.imshow(im_, cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")
        ax.scatter(le[ij[:, 1]], lx[ij[:, 0]], c=v, cmap="viridis", vmin=0, vmax=1, s=28, edgecolors="w",
                   linewidths=0.9)
        ax.set_title("%s — %d puntos KMC" % (t, len(ij)))
        ax.set_xlabel(r"$\log(\ell)$")
        ax.set_ylabel(r"$\log(\Xi)$")
    fig.colorbar(im, ax=axs, label="SoC", shrink=0.9)
    fig.savefig(os.path.join(a.out, "final.png"), dpi=140)
    print("->", os.path.join(a.out, "final.npy"), os.path.join(a.out, "final.png"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "fuse"):
        p = sp.add_parser(name)
        p.add_argument("--state", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "fuse": cmd_fuse}[a.cmd](a)


if __name__ == "__main__":
    main()
