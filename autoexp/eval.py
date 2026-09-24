"""Evaluador -- ARCHIVO FIJO, el loop de experimentacion no lo toca.

Uso:
    ./venv/bin/python -m autoexp.eval <run_dir> [--note "texto"] [--no-log]

`run_dir` tiene una subcarpeta por g (`g-4.0/`, ...) con:
    queries.json   registro del oraculo (lo escribe autoexp.oracle)
    restored.npy   reconstruccion; se acepta (H,W), (1,H,W) o con el pad de +1 px
                   de dip.restoration ((1,H+2,W+2)), que se recorta.

Verifica por g: consultas <= presupuesto, y que cada valor registrado coincide con
la imagen real (si no, el registro fue tocado a mano -> invalido).

Metricas por g (contra sim_128.png en [0,1], 128x128):
    psnr        PSNR imagen completa (data_range=1)
    ssim        SSIM imagen completa
    psnr_front  PSNR solo en la banda de frontera (pixeles con |grad| de la imagen
                real en el 10% superior) -- la metrica que faltaba segun la
                presentacion del 17/9: la de imagen completa la dominan las mesetas.

Score de la corrida: `mean_psnr` sobre los g evaluados (lo que se optimiza), y
`min_psnr` (el peor g, siempre se reporta al lado). Objetivo del 23/9:
<= 64 consultas y > 38 dB.

Escribe `run_dir/score.json`, `run_dir/panel.png` (original | puntos | reconstruida
| |error| por g) y agrega una fila a `autoexp/leaderboard.csv`.
"""
import argparse
import csv
import datetime
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from autoexp.oracle import load_truth

HERE = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD = os.path.join(HERE, "leaderboard.csv")
TARGET_PTS, TARGET_DB = 64, 38.0


def _load_restored(path, shape):
    r = np.load(path).astype(np.float64)
    if r.ndim == 3:
        r = r[0]
    H, W = shape
    if r.shape == (H + 2, W + 2):
        r = r[1:-1, 1:-1]
    if r.shape != (H, W):
        raise ValueError("%s: shape %s, se esperaba %s" % (path, r.shape, shape))
    return np.clip(r, 0.0, 1.0)


def _front_band(truth, q=0.90):
    gy, gx = np.gradient(truth)
    gm = np.hypot(gx, gy)
    return gm >= np.quantile(gm, q)


def evaluate_g(gdir):
    with open(os.path.join(gdir, "queries.json")) as f:
        qlog = json.load(f)
    g = qlog["g"]
    truth = load_truth(g)
    pts = np.array([p[:2] for p in qlog["points"]], dtype=int).reshape(-1, 2)
    vals = np.array([p[2] for p in qlog["points"]])
    n = len(pts)
    if n > qlog["budget"]:
        raise ValueError("%s: %d consultas > presupuesto %d" % (gdir, n, qlog["budget"]))
    if len({tuple(p) for p in pts}) != n:
        raise ValueError("%s: consultas duplicadas en el registro" % gdir)
    if n and not np.allclose(truth[pts[:, 0], pts[:, 1]], vals, atol=1e-9):
        raise ValueError("%s: el registro de consultas no coincide con la imagen real" % gdir)

    rec = _load_restored(os.path.join(gdir, "restored.npy"), truth.shape)
    band = _front_band(truth)
    err = rec - truth
    res = {
        "g": g,
        "n_points": n,
        "psnr": float(peak_signal_noise_ratio(truth, rec, data_range=1.0)),
        "ssim": float(structural_similarity(truth, rec, data_range=1.0)),
        "psnr_front": float(10 * np.log10(1.0 / max(np.mean(err[band] ** 2), 1e-12))),
        "mae": float(np.abs(err).mean()),
    }
    return res, truth, rec, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    gdirs = sorted(glob.glob(os.path.join(a.run_dir, "g*")), key=lambda d: float(os.path.basename(d)[1:]))
    gdirs = [d for d in gdirs if os.path.isfile(os.path.join(d, "restored.npy"))]
    if not gdirs:
        raise SystemExit("no hay g*/restored.npy en %s" % a.run_dir)

    rows, figs = [], []
    for d in gdirs:
        r, truth, rec, pts = evaluate_g(d)
        rows.append(r)
        figs.append((r, truth, rec, pts))

    ps = [r["psnr"] for r in rows]
    summary = {
        "run": os.path.basename(os.path.normpath(a.run_dir)),
        "run_dir": os.path.abspath(a.run_dir),
        "mean_psnr": float(np.mean(ps)),
        "min_psnr": float(np.min(ps)),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "mean_psnr_front": float(np.mean([r["psnr_front"] for r in rows])),
        "max_points": int(max(r["n_points"] for r in rows)),
        "per_g": rows,
    }
    summary["goal_met"] = bool(summary["max_points"] <= TARGET_PTS and summary["min_psnr"] > TARGET_DB)
    with open(os.path.join(a.run_dir, "score.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # panel: una fila por g
    n = len(figs)
    fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for k, (r, truth, rec, pts) in enumerate(figs):
        ax[k, 0].imshow(truth, cmap="viridis", vmin=0, vmax=1)
        ax[k, 0].set_ylabel("g = %s" % r["g"])
        ax[k, 1].imshow(np.zeros_like(truth), cmap="gray", vmin=0, vmax=1)
        if len(pts):
            ax[k, 1].scatter(pts[:, 1], pts[:, 0], c=truth[pts[:, 0], pts[:, 1]], cmap="viridis",
                             vmin=0, vmax=1, s=14, edgecolors="w", linewidths=0.3)
        ax[k, 2].imshow(rec, cmap="viridis", vmin=0, vmax=1)
        ax[k, 3].imshow(np.abs(rec - truth), cmap="magma", vmin=0, vmax=0.5)
        ax[k, 1].set_title("%d puntos consultados" % r["n_points"], fontsize=9)
        ax[k, 2].set_title("PSNR %.1f dB  SSIM %.3f" % (r["psnr"], r["ssim"]), fontsize=9)
        ax[k, 3].set_title("|error|  (frontera %.1f dB)" % r["psnr_front"], fontsize=9)
        for c in range(4):
            ax[k, c].set_xticks([])
            ax[k, c].set_yticks([])
    for c, t in enumerate(["Original", "Puntos", "Reconstruida", "|error|"]):
        ax[0, c].set_xlabel(t)
        ax[0, c].xaxis.set_label_position("top")
    fig.suptitle("%s  --  media %.1f dB, peor g %.1f dB, <= %d puntos"
                 % (summary["run"], summary["mean_psnr"], summary["min_psnr"], summary["max_points"]))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3 * n)))  # deja lugar al suptitle
    fig.savefig(os.path.join(a.run_dir, "panel.png"), dpi=110)
    plt.close(fig)

    if not a.no_log:
        new = not os.path.exists(LEADERBOARD)
        with open(LEADERBOARD, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["fecha", "run", "g_evaluados", "max_puntos", "mean_psnr", "min_psnr",
                            "mean_ssim", "mean_psnr_front", "objetivo", "nota", "run_dir"])
            w.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), summary["run"],
                " ".join(str(r["g"]) for r in rows), summary["max_points"],
                "%.2f" % summary["mean_psnr"], "%.2f" % summary["min_psnr"],
                "%.4f" % summary["mean_ssim"], "%.2f" % summary["mean_psnr_front"],
                "SI" if summary["goal_met"] else "no", a.note, os.path.relpath(a.run_dir, os.path.dirname(HERE)),
            ])

    print("%-40s mean %.2f dB | min %.2f dB | SSIM %.4f | frontera %.2f dB | <=%d pts | objetivo: %s"
          % (summary["run"], summary["mean_psnr"], summary["min_psnr"], summary["mean_ssim"],
             summary["mean_psnr_front"], summary["max_points"], "SI" if summary["goal_met"] else "no"))
    for r in rows:
        print("   g=%-5s  %2d pts  PSNR %6.2f  SSIM %.4f  frontera %6.2f"
              % (r["g"], r["n_points"], r["psnr"], r["ssim"], r["psnr_front"]))


if __name__ == "__main__":
    main()
