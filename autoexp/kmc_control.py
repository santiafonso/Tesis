"""Tanda de control de la reconstruccion KMC: puntos al azar que NO se usaron para reconstruir;
se corren con KMC y se comparan con el mapa (la unica medida de error real con KMC, que no
tiene mapa completo contra el cual calcular PSNR).

    python -m autoexp.kmc_control pick --state kmc.json --out control.csv [--n 10] [--min-logell -3]
    python -m autoexp.kmc_control eval --state kmc.json --results resultados_control.csv
                                       --final dip/final.npy [--out control]  -> control.png / control.json

Puntos (5/10, pedido de los profes: mas de 10 y en los lugares criticos): --n (def 32), de los
cuales --frac-crit (def 0.75) son los sorteados con peso puntaje^2, con puntaje critico = |reconstruccion - continuo| (donde
el KMC se aparta del modelo) + |grad reconstruccion| (frente de fase), ambos normalizados a [0, 1];
el resto se sortea uniforme en todo el mapa (semilla fija), para que el error no quede sesgado.
Todos lejos (>= --min-dist px) de los ya medidos y entre si, y con log l >= --min-logell (def -2.7,
<8 h por corrida en g=-4): mas abajo no termina en el tope de 8 h. `critico.png` muestra el puntaje y los
puntos. Con --frac-crit 0 es el sorteo viejo (10 puntos, 70 % en SoC > 0.05).
"""
import argparse
import csv
import json
import os

import numpy as np

from autoexp import interp, kmc_planner
from autoexp.oracle import load_truth


def critical_score(st, rec):
    """|rec - continuo| + |grad rec|, cada uno normalizado a [0, 1]; sin continuo, solo el gradiente."""
    gy, gx = np.gradient(rec)
    grad = np.hypot(gx, gy)
    score = grad / (grad.max() or 1)
    try:
        cont = load_truth("%.1f" % st.get("g", -4.0))
        diff = np.abs(rec - cont)
        score = score + diff / (diff.max() or 1)
    except FileNotFoundError:
        print("sin continuo para g = %s: puntaje solo por gradiente" % st.get("g"))
    return score


def cmd_pick(a):
    st = kmc_planner.load(a.state)
    lx, le = kmc_planner.axes(st)
    known = np.array(list(st["known_d"].keys()))
    rec, _ = kmc_planner.reconstruct(st)
    rng = np.random.default_rng(a.seed)
    pts = []

    def ok(i, j):
        return (le[j] >= a.min_logell and np.min(np.hypot(*(known - [i, j]).T)) >= a.min_dist
                and all(np.hypot(i - p, j - q) >= a.min_dist for p, q in pts))

    score = critical_score(st, rec)
    n_crit = int(round(a.n * a.frac_crit))
    w = (score ** 2).ravel()  # sorteo con peso puntaje^2: cubre todo el frente, no solo el maximo
    for flat in rng.choice(w.size, size=int((w > 0).sum()), replace=False, p=w / w.sum()):
        if len(pts) >= n_crit:
            break
        i, j = np.unravel_index(flat, score.shape)
        if ok(i, j):
            pts.append((int(i), int(j)))
    n_int = len(pts) + int(round((a.n - len(pts)) * 0.7)) if not a.frac_crit else 0
    while len(pts) < a.n:
        i, j = rng.integers(0, st["res"], 2)
        if (len(pts) < n_int and rec[i, j] <= 0.05) or not ok(i, j):
            continue
        pts.append((int(i), int(j)))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        gcol = ["%g" % st["g"]] if "g" in st else []
        w.writerow(["row", "col", "logxi", "logell", "xi", "ell"] + (["g"] if gcol else []))
        for i, j in pts:
            w.writerow([i, j, "%.6f" % lx[i], "%.6f" % le[j], "%.6g" % 10 ** lx[i], "%.6g" % 10 ** le[j]] + gcol)
    # tipo de cada punto aparte (kmc_tanda.slurm lee las columnas por posicion: no se agrega al CSV)
    json.dump({"critico": pts[:n_crit], "uniforme": pts[n_crit:]}, open(a.out.replace(".csv", "_tipos.json"), "w"))
    print("%d puntos de control (%d criticos) -> %s" % (len(pts), n_crit, a.out))
    plot_pick(st, score, known, pts, n_crit, a.out.replace(".csv", "_critico.png"))


def plot_pick(st, score, known, pts, n_crit, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lx, le = kmc_planner.axes(st)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(score, cmap="magma", extent=[le[0], le[-1], lx[-1], lx[0]], aspect="auto")
    fig.colorbar(im, ax=ax, label="puntaje critico (|rec - continuo| + |grad|)")
    ax.scatter(le[known[:, 1]], lx[known[:, 0]], s=10, c="w", label="medidos")
    p = np.array(pts)
    ax.scatter(le[p[:n_crit, 1]], lx[p[:n_crit, 0]], s=40, marker="x", c="c", label="control critico")
    ax.scatter(le[p[n_crit:, 1]], lx[p[n_crit:, 0]], s=40, marker="+", c="lime", label="control uniforme")
    ax.set_xlabel("log l")
    ax.set_ylabel("log Xi")
    ax.set_title("Control g = %g: %d puntos" % (st.get("g", -4.0), len(pts)))
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(out, dpi=150)
    print("->", out)


def cmd_eval(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = kmc_planner.load(a.state)
    final = np.load(a.final)
    tps, _ = kmc_planner.reconstruct(st)
    ij = kmc_planner.split(st)[0][0]  # solo los medidos en la figura
    rows = [r for r in csv.DictReader(open(a.results)) if r["status"] == "terminado"]
    c = np.array([[int(r["row"]), int(r["col"])] for r in rows])
    kmc = np.array([float(r["value"]) for r in rows])
    out = {"n": len(rows), "sigma_kmc": st.get("sigma")}

    def stats(e):
        rms = float(np.sqrt(np.mean(e ** 2)))
        return {"n": len(e), "rms": rms, "mae": float(np.mean(np.abs(e))), "max": float(np.max(np.abs(e))),
                "psnr_equiv": float(20 * np.log10(1.0 / rms)) if rms > 0 else None}

    for name, m in (("fusion", final), ("tps", tps)):
        out[name] = stats(m[c[:, 0], c[:, 1]] - kmc)
    tipos_path = os.path.join(os.path.dirname(a.results), "control_tipos.json")
    tipo = np.array(["control"] * len(c), dtype=object)
    if os.path.exists(tipos_path):  # error aparte en los criticos y en los sorteados al azar
        tipos = json.load(open(tipos_path))
        for t in ("critico", "uniforme"):
            sel = np.array([[i, j] in tipos[t] for i, j in c.tolist()], bool)
            tipo[sel] = t
            if sel.any():
                out["fusion_" + t] = stats(final[c[sel, 0], c[sel, 1]] - kmc[sel])
    json.dump(out, open(a.out + ".json", "w"), indent=1)

    lx, le = kmc_planner.axes(st)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    im = axs[0].imshow(final, cmap="viridis", vmin=0, vmax=1, extent=[le[0], le[-1], lx[-1], lx[0]], aspect="auto")
    axs[0].scatter(le[ij[:, 1]], lx[ij[:, 0]], s=10, c="w", alpha=0.6, label="usados (%d)" % len(ij))
    for t, ec in (("control", "r"), ("critico", "r"), ("uniforme", "orange")):
        sel = tipo == t
        if sel.any():
            axs[0].scatter(le[c[sel, 1]], lx[c[sel, 0]], s=60, c=kmc[sel], cmap="viridis", vmin=0, vmax=1,
                           marker="s", edgecolors=ec, linewidths=1.5, label="%s (%d)" % (t, sel.sum()))
    axs[0].legend(loc="lower left", fontsize=8)
    axs[0].set_xlabel(r"$\log(\ell)$")
    axs[0].set_ylabel(r"$\log(\Xi)$")
    axs[0].set_title("Mapa final y puntos de control")
    fig.colorbar(im, ax=axs[0], label="SoC")
    pred = final[c[:, 0], c[:, 1]]
    axs[1].plot([0, 1], [0, 1], c="0.6", lw=1)
    for t, col in (("control", "C0"), ("critico", "C3"), ("uniforme", "C1")):
        sel = tipo == t
        if sel.any():
            r = out.get("fusion_" + t, out["fusion"])
            axs[1].scatter(kmc[sel], pred[sel], s=40, c=col, label="%s: RMS %.3f" % (t, r["rms"]))
    axs[1].legend(loc="upper left", fontsize=8)
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
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--frac-crit", type=float, default=0.75)
    p.add_argument("--min-dist", type=float, default=4)
    p.add_argument("--min-logell", type=float, default=-2.7)
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
