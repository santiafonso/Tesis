#!/usr/bin/env python
# coding: utf-8
"""Resumen de TODO lo probado en la sesion del 15/9 (fuera de los 5 puntos
oficiales tambien): PSNR final vs. cantidad de puntos REALES usados, para
cada g con datos. Junta:
  - el barrido original de 5 familias (reg_noise=0.01 default)
  - el barrido REG_NOISE_STD x familia x g (cluster, 328 pts)
  - el barrido de LR (82 pts, g=-4.0)
  - los experimentos de ventana (g=-4.0)
  - el experimento de puntos sinteticos de meseta (synth_augment, g=-4.0)

Uso:
    ./venv/bin/python -m analysis.all_experiments_summary

Escribe:
    results/experimentos_menos_puntos/all_experiments_summary.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GS_PLOT = ["-4.0", "-2.0", "-0.5"]

CATEGORY_COLOR = {
    "family_sweep": "#c9c9c9",
    "reg_sweep": "#1f77b4",
    "LR_sweep": "#9467bd",
    "ventana": "#ff7f0e",
    "synth_augment": "#d62728",
    "baseline": "#7f7f7f",
}
CATEGORY_LABEL = {
    "family_sweep": "barrido 5 familias (reg=0.01)",
    "reg_sweep": "familia x REG_NOISE_STD (cluster)",
    "LR_sweep": "barrido de LR (82 pts)",
    "ventana": "ventanas concentradas",
    "synth_augment": "puntos sinteticos de meseta",
    "baseline": "baseline puntual",
}


def categorize(method):
    if method.startswith("family_sweep"):
        return "family_sweep"
    if method.startswith("reg0."):
        return "reg_sweep"
    if method.startswith("LR_sweep"):
        return "LR_sweep"
    if method.startswith("ventana"):
        return "ventana"
    if method.startswith("synth_augment"):
        return "synth_augment"
    return "baseline"


def collect():
    rows = []
    with open("results/gsweep_5family_summary/summary.csv") as f:
        for r in csv.DictReader(f):
            pct = float(r["pct_obs"])
            n = round(pct / 100 * 128 * 128)
            rows.append(dict(g=r["g"], n=n, psnr=float(r["psnr_final"]),
                              method="family_sweep/" + r["family"]))

    with open("results/experimentos_menos_puntos/barrido_reg_noise_familias/summary.csv") as f:
        for r in csv.DictReader(f):
            rows.append(dict(g=r["g"], n=328, psnr=float(r["psnr"]),
                              method="reg%s/%s" % (r["reg"], r["family"])))

    lr_dir = "results/experimentos_menos_puntos/lr_regnoise_sweep/g-4.0/mf0.995/grid"
    if os.path.isdir(lr_dir):
        for d in os.listdir(lr_dir):
            fp = os.path.join(lr_dir, d, "metrics.csv")
            if os.path.isfile(fp):
                for row in csv.reader(open(fp)):
                    if row and row[0] == "final":
                        rows.append(dict(g="-4.0", n=82, psnr=float(row[1]), method="LR_sweep/" + d))

    fp = "results/dip_gsweep_frontier/g-4.0/mf0.995/grid/metrics.csv"
    for row in csv.reader(open(fp)):
        if row and row[0] == "final":
            rows.append(dict(g="-4.0", n=82, psnr=float(row[1]), method="baseline/grid_mf0.995"))

    for n, psnr, name in [
        (90, 24.8, "ventana/franja"),
        (90, 24.2, "ventana/64_reg0.08"),
        (121, 22.9, "ventana/64_concentrado"),
        (36, 18.6, "ventana/32_extrapol"),
    ]:
        rows.append(dict(g="-4.0", n=n, psnr=psnr, method=name))

    rows.append(dict(g="-4.0", n=82, psnr=11.06, method="synth_augment/82real_solo"))
    rows.append(dict(g="-4.0", n=482, psnr=24.16, method="synth_augment/82real+400synth"))
    rows.append(dict(g="-4.0", n=270, psnr=9.14, method="synth_augment/270real_busqueda_activa_solo"))
    rows.append(dict(g="-4.0", n=670, psnr=28.56, method="synth_augment/270real_busqueda_activa+400synth"))
    rows.append(dict(g="-4.0", n=481, psnr=25.61, method="synth_augment/82real_frontera_grilla_meseta"))
    rows.append(dict(g="-4.0", n=716, psnr=29.52, method="synth_augment/320real_frontera_grilla_meseta"))

    return rows


def main():
    rows = collect()
    fig, axs = plt.subplots(1, len(GS_PLOT), figsize=(16, 5.5), sharey=True)
    for ax, g in zip(axs, GS_PLOT):
        sub = [r for r in rows if r["g"] == g]
        cats_seen = set()
        for r in sub:
            cat = categorize(r["method"])
            label = CATEGORY_LABEL[cat] if cat not in cats_seen else None
            cats_seen.add(cat)
            ax.scatter(r["n"], r["psnr"], c=CATEGORY_COLOR[cat], s=45,
                       edgecolors="k", linewidths=0.3, alpha=0.85, label=label)
        # pareto frontier (mejor PSNR para N <= x)
        sub_sorted = sorted(sub, key=lambda r: r["n"])
        best_so_far = -1
        px, py = [], []
        for r in sub_sorted:
            if r["psnr"] > best_so_far:
                best_so_far = r["psnr"]
                px.append(r["n"])
                py.append(r["psnr"])
        ax.plot(px, py, color="black", linewidth=1.2, linestyle="--", alpha=0.6, zorder=0)
        ax.set_xscale("log")
        ax.set_title("g = %s" % g)
        ax.set_xlabel("# puntos reales usados")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("PSNR final (dB)")
    axs[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Todo lo probado el 15/9: PSNR vs. cantidad de puntos reales, por g\n"
        "(linea punteada = mejor PSNR alcanzado hasta ese N -- la frontera de Pareto)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = "results/experimentos_menos_puntos/all_experiments_summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("Escrito", out)


if __name__ == "__main__":
    main()
