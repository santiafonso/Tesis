"""Fusion DIP + reconstruccion clasica por distancia al acantilado -- MODIFICABLE.

final = monotone_2d( w * clasica + (1 - w) * DIP ),  w = exp(-(d / B)^2)

d = distancia (px) por encima del acantilado ajustado. Cerca del borde manda la clasica
(TPS con rampa alineada: precisa en frentes abruptos, donde DIP deja hueco o escalones);
lejos manda DIP (liso en mesetas y en la zona que se desvanece, donde la TPS ondula).
Debajo del borde ambas valen 0.

Se aplica sobre una corrida DIP ya hecha (en el cluster o traida con rsync): no hace falta
volver a correr DIP. La clasica se recalcula con los mismos puntos (queries.json).

Guarda (--guard): si DIP no reproduce los puntos REALES consultados de arriba del acantilado
(RMS del residuo > guard), DIP fallo en ese g -> se usa solo la clasica. Sin mirar el mapa:
en los 9 g, cuando DIP anda el residuo es <= 0.008 y cuando falla es >= 0.03.

Uso:
    python -m autoexp.fuse <run_dip_dir> <nombre_nuevo> [--B 10] [--guard 0.02] [--note ...]
Escribe autoexp/runs/<nombre_nuevo>/g*/{queries.json,restored.npy} y corre autoexp.eval.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from autoexp import interp

HERE = os.path.dirname(os.path.abspath(__file__))


def fuse_one(dip, classic, coef, B):
    H, W = classic.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.clip(np.polyval(coef, xx) - yy, 0, None)
    w = np.exp(-(d / B) ** 2)
    return interp.monotone_2d(np.clip(w * classic + (1 - w) * dip, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("name")
    ap.add_argument("--B", type=float, default=10.0)
    ap.add_argument("--guard", type=float, default=0.0, help="0 = sin guarda")
    ap.add_argument("--recon-kw", default="{}", help="kwargs de interp.cliff (ej. umbrales con ruido)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    dst = os.path.join(HERE, "runs", a.name)
    for gd in sorted(glob.glob(os.path.join(a.src, "g*"))):
        if not os.path.isfile(os.path.join(gd, "restored.npy")):
            continue
        out = os.path.join(dst, os.path.basename(gd))
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(gd, "queries.json"), out)
        pts = np.array(json.load(open(os.path.join(gd, "queries.json")))["points"])
        classic, info = interp.cliff(pts[:, :2], pts[:, 2], (128, 128), **json.loads(a.recon_kw))
        d = np.load(os.path.join(gd, "restored.npy")).astype(float)
        d = d[0] if d.ndim == 3 else d
        d = np.clip(d[1:-1, 1:-1] if d.shape[0] == classic.shape[0] + 2 else d, 0, 1)
        use_dip = info is not None
        if use_dip and a.guard > 0:
            ij = pts[:, :2].astype(int)
            up = (np.polyval(info[0], ij[:, 1]) - ij[:, 0]) > 0
            res = float(np.sqrt(np.mean((d[ij[up, 0], ij[up, 1]] - pts[up, 2]) ** 2)))
            use_dip = res <= a.guard
            print("   %s: residuo DIP en puntos reales %.4f -> %s" % (os.path.basename(gd), res,
                                                                      "fusion" if use_dip else "solo clasica"))
        rec = fuse_one(d, classic, info[0], a.B) if use_dip else classic
        np.save(os.path.join(out, "restored.npy"), rec)
    subprocess.run([sys.executable, "-m", "autoexp.eval", dst, "--note",
                    a.note or "fusion B=%g de %s" % (a.B, a.src)], check=True)


if __name__ == "__main__":
    main()
