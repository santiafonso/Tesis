#!/usr/bin/env python
# coding: utf-8
"""Ordena carpetas de salida de dip.restoration YA generadas al layout nuevo:
los `iter_*.png` sueltos (decenas por corrida, "todos iguales") pasan a un
subdir `snapshots/`, y se arma `snapshots_contact.png` con la trayectoria en
una sola imagen.

Idempotente y no destructivo: solo mueve PNGs de iteracion y crea el contact
sheet; si ya esta hecho, no toca nada.

Uso:
    ./venv/bin/python analysis/tidy_run_dirs.py results/dip_gsweep archive/dip_gsweep
    ./venv/bin/python analysis/tidy_run_dirs.py            # default: results/
"""
import glob
import os
import re
import sys

from PIL import Image

ITER_RE = re.compile(r"iter_\d+\.png$")


def contact_sheet(snap_dir, out_path, ncol=6):
    snaps = sorted(glob.glob(os.path.join(snap_dir, "iter_*.png")))
    if not snaps:
        return
    thumbs = [Image.open(p).convert("L") for p in snaps]
    w, h = thumbs[0].size
    ncol = min(ncol, len(thumbs))
    nrow = (len(thumbs) + ncol - 1) // ncol
    sheet = Image.new("L", (ncol * w, nrow * h), color=255)
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % ncol) * w, (k // ncol) * h))
    sheet.save(out_path)


def tidy(run_dir):
    iters = [
        f
        for f in os.listdir(run_dir)
        if ITER_RE.search(f) and os.path.isfile(os.path.join(run_dir, f))
    ]
    snap_dir = os.path.join(run_dir, "snapshots")
    if not iters and not os.path.isdir(snap_dir):
        return False
    os.makedirs(snap_dir, exist_ok=True)
    for f in iters:
        os.replace(os.path.join(run_dir, f), os.path.join(snap_dir, f))
    contact = os.path.join(run_dir, "snapshots_contact.png")
    if not os.path.exists(contact):
        contact_sheet(snap_dir, contact)
    if iters:
        print("  %s  (%d snapshots movidos)" % (run_dir, len(iters)))
    return True


def main(roots):
    n = 0
    for root in roots:
        if not os.path.isdir(root):
            print("!! no existe:", root)
            continue
        # una "run dir" es cualquier carpeta que tenga metrics.csv o restored.npy
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) == "snapshots":
                continue
            if "metrics.csv" in files or "restored.npy" in files or any(
                ITER_RE.search(f) for f in files
            ):
                if tidy(dirpath):
                    n += 1
    print("Listo. %d carpetas ordenadas." % n)


if __name__ == "__main__":
    main(sys.argv[1:] or ["results"])
