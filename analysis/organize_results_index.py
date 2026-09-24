#!/usr/bin/env python
# coding: utf-8
"""Indice navegable (symlinks, no copia datos) de las corridas de
results/experimentos_menos_puntos/, organizado como
<familia>/reg<valor>/<N>pts/g<valor>.png en vez de tener que recordar en
que carpeta de que experimento quedo cada combinacion.

Uso:
    ./venv/bin/python -m analysis.organize_results_index

Escribe (recreando desde cero cada vez, asi nunca queda un symlink viejo
colgado):
    results/experimentos_menos_puntos/organizado_por_familia_reg_puntos/
"""
import os
import shutil

ROOT = "results/experimentos_menos_puntos"
ORG = os.path.join(ROOT, "organizado_por_familia_reg_puntos")

GS = ["-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0"]
FAMILIES = ["grid", "uniform", "frontier_mix"]
REGS = ["0.01", "0.03", "0.08"]
N_BY_MF = {"0.980": "328pts", "0.990": "164pts", "0.995": "82pts"}


def collect():
    sources = []  # (family, reg, npts_label, g, src_dir)

    base1 = os.path.join(ROOT, "barrido_reg_noise_familias")
    for g in GS:
        for mf, nlabel in N_BY_MF.items():
            for fam in FAMILIES:
                for reg in REGS:
                    d = os.path.join(base1, "g%s" % g, "mf%s" % mf, fam, "reg%s" % reg)
                    if os.path.isfile(os.path.join(d, "comparison_annotated.png")):
                        sources.append((fam, reg, nlabel, g, os.path.abspath(d)))

    base2 = os.path.join(ROOT, "fewpoints_3g")
    for g in GS:
        d = os.path.join(base2, "g%s" % g, "mf0.995", "frontier_mix")
        if os.path.isfile(os.path.join(d, "comparison_annotated.png")):
            sources.append(("frontier_mix", "0.08", "82pts", g, os.path.abspath(d)))

    return sources


def main():
    if os.path.isdir(ORG):
        shutil.rmtree(ORG)
    os.makedirs(ORG)

    sources = collect()
    for fam, reg, nlabel, g, src in sources:
        dest_dir = os.path.join(ORG, fam, "reg%s" % reg, nlabel)
        os.makedirs(dest_dir, exist_ok=True)
        os.symlink(os.path.join(src, "comparison_annotated.png"), os.path.join(dest_dir, "g%s.png" % g))

    print("Symlinks creados: %d en %s" % (len(sources), ORG))


if __name__ == "__main__":
    main()
