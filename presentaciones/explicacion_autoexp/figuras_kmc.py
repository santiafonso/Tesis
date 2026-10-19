#!/usr/bin/env python
# coding: utf-8
"""Figuras de explicacion_autoexp que no salen del caso g = -2 de figuras.py:

- tps_que_es.png: que es la TPS (1D y el nucleo r^2 log r).
- tps_continuo_g.png: la receta en los 11 g del continuo (incl. g = +2, +4) con ruido tipo KMC.
- kmc_tandas_g-4.png: el mapa de g = -4 (KMC real) reconstruido despues de cada tanda.
- kmc_pasos_g-4.png: los 8 pasos de la reconstruccion final de g = -4 (puntos -> TPS -> borde ->
  cero abajo -> monotonia -> DIP -> peso -> fusion).

Las usa make_explicacion_autoexp.py.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from autoexp import interp, kmc_planner  # noqa: E402
from autoexp.oracle import Oracle, load_truth  # noqa: E402
from figuras import C1, C2, FIG, INK2, _clean, _psnr, _save  # noqa: E402

KMC = os.path.join(REPO, "autoexp", "kmc", "g-4")
GS = ["-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0", "2.0", "4.0"]
SIGMA = 0.005  # ruido medido entre replicas de KMC (0.002-0.006)
B = 20.0  # ancho de la fusion, el de kmc_dip

# epigrafes de cada cuadro de las secuencias (los usan las dos presentaciones)
CAP_TANDAS = ["Tanda 1: grilla de 30. Sin pares 0 / alto cercanos todavía: TPS común, el frente sale ondulado.",
         "Tanda 2: primera bisección (círculos naranjas = puntos nuevos). Todavía no alcanza para ubicar el borde.",
         "Tanda 3: la bisección se acerca al salto desde los dos lados.",
         "Tanda 4: aparecen pares 0 / alto a ≤ 4 px → se detecta el borde (línea punteada) y debajo pasa a 0.",
         "Tanda 5: el borde se afina; la rampa sobre el borde ya tiene forma.",
         "Tanda 6: termina la bisección (54 puntos).",
         "Tanda 7: relleno, donde la validación cruzada dice que más se equivoca (arriba a la derecha).",
         "Tanda 8: más relleno en la zona que se desvanece.",
         "Tanda 9: 64 puntos. Esto es la TPS con restricciones físicas.",
         "Final: se mezcla con DIP lejos del borde (fusión con guarda)."]
CAP_PASOS = ["1. Los 64 valores de SoC del KMC: lo único que hay.",
         "2. TPS sola (scipy, thin-plate spline): se pasa de 0 y de 1 y ondula en el frente. Sola no sirve.",
         "3. Borde: aristas cortas de Delaunay entre un punto ≈ 0 y uno alto; sus puntos medios (★) → recta. "
         "El salto es real: a 1 px, el KMC da 0.77 y 0.00.",
         "4. Debajo de la recta, 0. Arriba, TPS solo con los puntos de arriba, alineada al borde.",
         "5. + cotas y proyección monótona (el SoC no sube con ℓ ni con Ξ) = TPS con restricciones físicas.",
         "6. DIP: la red ve los puntos + ceros debajo del borde; 2 semillas, queda la que mejor ajusta los puntos.",
         "7. Peso de la mezcla: 1 pegado al borde (TPS), 0 lejos (DIP).",
         "8. Final = monótona(w·TPS + (1−w)·DIP). Si DIP no reproduce los puntos, queda solo la TPS."]


def _frames(fig, axs, n, name):
    """Guarda name_1.png ... name_n.png con los primeros k paneles visibles (el resto en blanco,
    mismo layout): pasando las filminas se ve la reconstruccion armarse paso a paso."""
    vis = [a.get_visible() for a in axs]
    for k in range(1, n + 1):
        for i, a in enumerate(axs):
            a.set_visible(vis[i] and i < k)
        _save_keep(fig, "%s_%d.png" % (name, k))
    for a, v_ in zip(axs, vis):
        a.set_visible(v_)


def _save_keep(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150)
    return p


def _kmc():
    st = kmc_planner.load(os.path.join(KMC, "kmc.json"))
    lx, le = kmc_planner.axes(st)
    return st, lx, le


def _ext(lx, le):
    return [le[0], le[-1], lx[-1], lx[0]]


def fig_tps_que_es():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.array([0, 1, 2.2, 3, 4.5, 6.0])
    y = np.array([1, 0.95, 0.9, 0.2, 0, 0])
    xx = np.linspace(-0.3, 6.3, 400)
    axs[0].plot(xx, np.interp(xx, x, y), color="#9a9890", lw=1.2, label="lineal a tramos")
    axs[0].plot(xx, RBFInterpolator(x[:, None], y, kernel="thin_plate_spline")(xx[:, None]), color=C1, lw=2.4,
                label="TPS")
    axs[0].plot(xx, RBFInterpolator(x[:, None], y, kernel="thin_plate_spline", smoothing=0.5)(xx[:, None]),
                color=C2, lw=2, ls="--", label="TPS con suavizado (ruido)")
    axs[0].plot(x, y, "o", color="k", ms=7, zorder=5, label="puntos medidos")
    axs[0].axhline(0, color=INK2, lw=0.6)
    axs[0].set_title("La curva más lisa que pasa por los puntos\n(pero se pasa de largo en el salto)", fontsize=13)
    axs[0].legend(fontsize=10, loc="upper right")
    r = np.linspace(0.01, 1.5, 200)
    axs[1].plot(r, r ** 2 * np.log(r), color=C1, lw=2.4)
    axs[1].axhline(0, color=INK2, lw=0.6)
    axs[1].set_xlabel("distancia r al punto medido")
    axs[1].set_title(r"Cada punto aporta un bulto $\phi(r) = r^2 \log r$", fontsize=13)
    fig.tight_layout()
    return _save(fig, "tps_que_es.png")


def _simular(g):
    """64 puntos con el planificador de KMC sobre el mapa del continuo, con ruido SIGMA."""
    np.random.seed(0)
    orc = Oracle(g, budget=64, noise=SIGMA)
    st = {"res": 128, "budget": 64, "sigma": SIGMA, "sampler_kw": kmc_planner.SAMPLER_KW, "known_d": {},
          "rounds": []}
    while True:
        pts = kmc_planner.next_batch(st)
        if not pts:
            break
        st["known_d"].update({p: float(x) for p, x in zip(pts, orc.query(pts))})
    return np.array(list(st["known_d"].keys()), float), np.array(list(st["known_d"].values()))


def fig_continuo_g():
    """3 filas (verdad / TPS sola / TPS con restricciones fisicas) x 11 g."""
    fig, axs = plt.subplots(3, len(GS), figsize=(2.0 * len(GS), 6.9))
    out = []
    for c, g in enumerate(GS):
        truth = load_truth(g)
        ij, v = _simular(g)
        plain = interp.reconstruct(ij, v, truth.shape, "rbf_tps", smoothing=1e-4)
        rec, dec = interp.auto(ij, v, truth.shape, sigma=SIGMA)
        pp, pr = _psnr(truth, plain), _psnr(truth, rec)
        out.append((float(g), pp, pr, dec["cliff"]))
        axs[0, c].imshow(truth, cmap="viridis", vmin=0, vmax=1)
        axs[0, c].set_title("g = %s" % g.replace(".0", ""), fontsize=12)
        for r_, im, p in ((1, plain, pp), (2, rec, pr)):
            axs[r_, c].imshow(im, cmap="viridis", vmin=0, vmax=1)
            axs[r_, c].scatter(ij[:, 1], ij[:, 0], s=3, c="w", linewidths=0)
            axs[r_, c].set_title("%.1f dB%s" % (p, "" if r_ == 1 or dec["cliff"] else "\n(sin borde)"),
                                 fontsize=11, color=C2 if p < 38 else C1)
        for r_ in range(3):
            _clean(axs[r_, c])
    axs[0, 0].set_ylabel("verdad\n(continuo)", fontsize=11)
    axs[1, 0].set_ylabel("TPS sola", fontsize=11)
    axs[2, 0].set_ylabel("TPS con\nrestricciones\nfísicas", fontsize=11)
    fig.tight_layout()
    return _save(fig, "tps_continuo_g.png"), out


def _tandas(st):
    """Lista de (n_tanda, [(i, j), ...]) en orden, solo con los puntos ya cargados."""
    out = []
    for k in range(1, 20):
        p = os.path.join(KMC, "tanda_%02d.csv" % k)
        if not os.path.isfile(p):
            break
        pts = [(int(r["row"]), int(r["col"])) for r in csv.DictReader(open(p))]
        out.append((k, [q for q in pts if q in st["known_d"]]))
    seen = {q for _, ps in out for q in ps}
    resto = [q for q in st["known_d"] if q not in seen]  # pseudo que no estaban en ninguna tanda
    if resto:
        out[-1] = (out[-1][0], out[-1][1] + resto)
    return out


def fig_kmc_tandas():
    st, lx, le = _kmc()
    ext = _ext(lx, le)
    X = lambda j: np.interp(j, np.arange(len(le)), le)  # noqa: E731
    Y = lambda i: np.interp(i, np.arange(len(lx)), lx)  # noqa: E731
    final = np.load(os.path.join(KMC, "dip", "final.npy"))
    tandas = _tandas(st)
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.4))
    axs = axs.ravel()
    acc = []
    out = []
    for a, (k, ps) in zip(axs, tandas):
        nuevos = np.array(ps, float).reshape(-1, 2)
        acc += ps
        ij = np.array(acc, float)
        v = np.array([st["known_d"][q] for q in acc])
        rec, info = interp.cliff(ij, v, final.shape, sigma=st["sigma"])
        out.append((k, len(acc), info is not None))
        a.imshow(rec, cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")
        a.scatter(X(ij[:, 1]), Y(ij[:, 0]), s=10, c="w", linewidths=0)
        if len(nuevos):
            a.scatter(X(nuevos[:, 1]), Y(nuevos[:, 0]), s=40, facecolors="none", edgecolors=C2, linewidths=1.6)
        if info is not None:
            xs = np.arange(len(le))
            a.plot(X(xs), Y(np.clip(interp.edge(info[0], xs), 0, len(lx) - 1)), color="w", ls="--", lw=1)
        a.set_title("tanda %d: %d puntos%s" % (k, len(acc), "" if info is not None else "\n(sin borde todavía)"),
                    fontsize=12)
        _clean(a)
    a = axs[len(tandas)]
    a.imshow(final, cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")
    ij = np.array(acc, float)
    a.scatter(X(ij[:, 1]), Y(ij[:, 0]), s=10, c="w", linewidths=0)
    a.set_title("final: + DIP (fusión)", fontsize=12, color=C1)
    _clean(a)
    for a in axs[len(tandas) + 1:]:
        a.axis("off")
    fig.tight_layout()
    _frames(fig, axs, len(tandas) + 1, "kmc_tandas_g-4")
    return _save(fig, "kmc_tandas_g-4.png"), out


def fig_kmc_pasos():
    st, lx, le = _kmc()
    ext = _ext(lx, le)
    H = W = st["res"]
    sig = st["sigma"]
    X = lambda j: np.interp(j, np.arange(W), le)  # noqa: E731
    Y = lambda i: np.interp(i, np.arange(H), lx)  # noqa: E731
    ij, v = kmc_planner.recon_points(st)  # medidos + pseudo con valor max(cota, TPS de lo medido)
    ij = ij.astype(float)
    rk = kmc_planner.recon_kw(st)
    raw = RBFInterpolator(ij / H, v, kernel="thin_plate_spline", smoothing=1e-4)(
        np.mgrid[0:H, 0:W].reshape(2, -1).T / H).reshape(H, W)
    eps = interp.noise_params(sig).get("eps", 0.004)
    tri = Delaunay(ij)
    E = {tuple(sorted((s[a], s[b]))) for s in tri.simplices for a, b in ((0, 1), (1, 2), (0, 2))}
    cut = [(a, b) for a, b in E if (v[a] < eps) != (v[b] < eps) and np.linalg.norm(ij[a] - ij[b]) <= 4]
    full, (coef, mids) = interp.cliff(ij, v, (H, W), **rk)
    nomono, _ = interp.cliff(ij, v, (H, W), sigma=sig, mono=False, bounds=False)
    dip = np.load(os.path.join(KMC, "dip", "mejor_de_2", "gkmc", "restored.npy"))
    final = np.load(os.path.join(KMC, "dip", "final.npy"))
    yy, xx = np.mgrid[0:H, 0:W]
    w = np.exp(-(np.clip(interp.edge(coef, xx) - yy, 0, None) / B) ** 2)
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    axs = axs.ravel()

    def show(a, im, t, pts=True, **kw):
        kw = dict(dict(cmap="viridis", vmin=0, vmax=1), **kw)
        a.imshow(im, extent=ext, aspect="auto", **kw)
        if pts:
            a.scatter(X(ij[:, 1]), Y(ij[:, 0]), c=v, cmap="viridis", vmin=0, vmax=1, s=18, edgecolors="w",
                      linewidths=0.7)
        a.set_title(t, fontsize=12)
        a.set_xlabel(r"$\log \ell$")
        a.set_ylabel(r"$\log \Xi$")

    axs[0].set_facecolor("#ececea")
    axs[0].scatter(X(ij[:, 1]), Y(ij[:, 0]), c=v, cmap="viridis", vmin=0, vmax=1, s=40, edgecolors="k",
                   linewidths=0.5)
    axs[0].set_xlim(ext[:2])
    axs[0].set_ylim(ext[2], ext[3])
    axs[0].set_title("1. Los %d puntos KMC" % len(ij), fontsize=12)
    show(axs[1], raw, "2. TPS sola (escala −0.3 a 1.3):\nse pasa de 0 y de 1", cmap="RdBu_r", vmin=-0.3, vmax=1.3)
    show(axs[2], interp.reconstruct(ij, v, (H, W), "rbf_tps", smoothing=1e-4),
         "3. Borde: aristas cortas 0 ↔ no-0,\npuntos medios (★) y recta")
    for a_, b_ in cut:
        axs[2].plot(X(ij[[a_, b_], 1]), Y(ij[[a_, b_], 0]), "r-", lw=1)
    axs[2].plot(X(mids[:, 1]), Y(mids[:, 0]), "r*", ms=11)
    xs = np.arange(W)
    axs[2].plot(X(xs), Y(np.clip(interp.edge(coef, xs), 0, H - 1)), "w--", lw=2)
    show(axs[3], nomono, "4. Cero debajo; arriba TPS solo con\nlos puntos de arriba, alineada al borde")
    show(axs[4], full, "5. + cotas y monotonía\n= TPS con restricciones físicas")
    show(axs[5], dip, "6. DIP (puntos + ceros debajo del borde),\nmejor de 2 semillas")
    show(axs[6], w, "7. Peso: w = exp(−(d/20)²)\n1 = manda la TPS, 0 = manda DIP", pts=False, cmap="magma")
    show(axs[7], final, "8. Final = monótona(w·TPS + (1−w)·DIP)")
    fig.tight_layout()
    _frames(fig, axs, 8, "kmc_pasos_g-4")
    return _save(fig, "kmc_pasos_g-4.png"), {"coef": [float(c) for c in coef], "n_mids": len(mids)}


def fig_scipy_metodos(g="-2.0"):
    """Las 4 interpolaciones de scipy (punto 6 de la reunion) sobre la misma grilla 8x8 de 64 puntos:
    reconstruccion arriba, |error| abajo."""
    from autoexp import sampling
    truth = load_truth(g)
    orc = Oracle(g, budget=64)
    sampling.grid(orc, 64, nx=8)
    ij, v = orc.observed()
    metodos = [("nearest", "vecino más cercano"), ("linear", "lineal"), ("cubic", "cúbica"), ("rbf_tps", "TPS")]
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.2))
    axs[0, 0].imshow(truth, cmap="viridis", vmin=0, vmax=1)
    axs[0, 0].scatter(ij[:, 1], ij[:, 0], s=14, c="w", edgecolors="k", linewidths=0.4)
    axs[0, 0].set_title("verdad + grilla 8×8\n(64 puntos)", fontsize=12.5)
    axs[1, 0].axis("off")
    out = []
    for k, (m, nombre) in enumerate(metodos, 1):
        rec = interp.reconstruct(ij, v, truth.shape, m)
        p = _psnr(truth, rec)
        out.append((nombre, p))
        axs[0, k].imshow(rec, cmap="viridis", vmin=0, vmax=1)
        axs[0, k].scatter(ij[:, 1], ij[:, 0], s=6, c="w", linewidths=0)
        axs[0, k].set_title("%s\n%.1f dB" % (nombre, p), fontsize=12.5, color=C2 if m == "rbf_tps" else INK2)
        axs[1, k].imshow(np.abs(rec - truth), cmap="magma", vmin=0, vmax=0.5)
        _clean(axs[1, k])
    for a in axs[0]:
        _clean(a)
    axs[0, 1].set_ylabel("reconstrucción", fontsize=12)
    axs[1, 1].set_ylabel("|error| (0 a 0.5)", fontsize=12)
    fig.tight_layout()
    return _save(fig, "scipy_metodos.png"), out


def main(continuo=True):
    fig_tps_que_es()
    _, pasos = fig_kmc_pasos()
    _, tandas = fig_kmc_tandas()
    _, scipy_m = fig_scipy_metodos()
    # el continuo tarda ~10 min (simula el planificador en 11 g): se reusa si ya esta calculado;
    # REHACER_CONTINUO=1 lo fuerza
    cache = os.path.join(FIG, "continuo.json")
    if continuo and (os.environ.get("REHACER_CONTINUO") or not os.path.isfile(cache)):
        _, cont = fig_continuo_g()
        json.dump([[float(x) for x in r[:3]] + [bool(r[3])] for r in cont], open(cache, "w"))
    cont = [tuple(r) for r in json.load(open(cache))] if os.path.isfile(cache) else None
    info = {"kmc_pasos": pasos, "kmc_tandas": tandas, "continuo": cont, "scipy_metodos": scipy_m}
    print(info)
    return info


if __name__ == "__main__":
    main()
