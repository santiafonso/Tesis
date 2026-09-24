"""Planificador de tandas de KMC con el muestreo activo de autoexp.

Cada punto del diagrama con KMC cuesta horas. Este script dice QUE puntos (log Xi, log l)
correr en cada tanda (en paralelo en el cluster), recibe los resultados, propone la tanda
siguiente, y al final reconstruye el mapa completo.

Como funciona: en cada `next` se vuelve a correr el muestreo (grilla -> biseccion del
acantilado por rondas -> relleno por validacion cruzada de a 4) desde cero con un oraculo de
"repeticion" que responde con los resultados ya cargados. Cuando el muestreo pide puntos que
todavia no se corrieron, se frena y esos puntos son la tanda siguiente. El muestreo es
determinista, asi que siempre llega al mismo lugar.

Uso:
    python -m autoexp.kmc_planner init  --state kmc.json [--res 128] [--budget 64] [--sigma 0.01]
                                        [--axes-npz results/.../soc_axes.npz]
    python -m autoexp.kmc_planner next  --state kmc.json [--out tanda.csv]
    python -m autoexp.kmc_planner add   --state kmc.json resultados.csv
    python -m autoexp.kmc_planner reconstruct --state kmc.json --out mapa   (-> mapa.npy, mapa.png)
    python -m autoexp.kmc_planner simulate --g -2.0 [--sigma 0.01]   (prueba de punta a punta
                                        contra un mapa del continuo, contando rondas)

resultados.csv: columnas `row,col,value` o `logxi,logell,value` (value = SoC_max en [0, 1]).
`sigma` = ruido por punto de KMC (estimarlo con corridas repetidas, NRUNS); fija los umbrales.

Convencion de ejes (la de dip.phase_diagram): fila 0 = log Xi mas alto (2), baja hasta -4;
columna 0 = log l mas bajo (-4), sube hasta 2. Con --axes-npz se usan los valores exactos.
"""
import argparse
import csv
import json
import os

import numpy as np

from autoexp import interp, sampling

SAMPLER_KW = {"n1": 30, "nx": 6, "target": "zero", "fill": "loo", "batch": 4}


class NeedPoints(Exception):
    def __init__(self, pts):
        super().__init__("faltan %d puntos" % len(pts))
        self.pts = pts


class ReplayOracle:
    """Misma interfaz que autoexp.oracle.Oracle, pero responde con resultados ya conocidos
    y corta con NeedPoints cuando piden algo que no se corrio."""

    def __init__(self, known, shape, budget):
        self.known = known  # (i, j) -> valor
        self.shape = shape
        self.budget = budget
        self.queried = {}

    @property
    def n_used(self):
        return len(self.queried)

    @property
    def remaining(self):
        return self.budget - self.n_used

    def query(self, points):
        pts = [(int(i), int(j)) for i, j in points]
        new = [p for p in dict.fromkeys(pts) if p not in self.queried]
        if self.n_used + len(new) > self.budget:
            raise RuntimeError("presupuesto excedido")
        missing = [p for p in new if p not in self.known]
        if missing:
            raise NeedPoints(missing)
        for p in new:
            self.queried[p] = self.known[p]
        return np.array([self.queried[p] for p in pts])

    def observed(self):
        if not self.queried:
            return np.zeros((0, 2), int), np.zeros(0)
        return np.array(list(self.queried.keys()), int), np.array(list(self.queried.values()))


# ------------------------------------------------------------------ ejes -------
def axes(st):
    n = st["res"]
    if st.get("axes"):
        return np.array(st["axes"]["logxi_rows"]), np.array(st["axes"]["logell_cols"])
    return np.linspace(2.0, -4.0, n), np.linspace(-4.0, 2.0, n)


def to_log(st, i, j):
    lx, le = axes(st)
    return float(lx[i]), float(le[j])


def to_pix(st, logxi, logell):
    lx, le = axes(st)
    return int(np.argmin(np.abs(lx - logxi))), int(np.argmin(np.abs(le - logell)))


# ----------------------------------------------------------------- estado ------
def load(path):
    st = json.load(open(path))
    st["known_d"] = {tuple(map(int, k.split(","))): v for k, v in st["known"].items()}
    return st


def save(st, path):
    st = dict(st)
    st["known"] = {"%d,%d" % k: v for k, v in st.pop("known_d").items()}
    json.dump(st, open(path, "w"), indent=1)


def next_batch(st):
    """Tanda siguiente (lista de (i, j)), o [] si el presupuesto ya se completo."""
    orc = ReplayOracle(st["known_d"], (st["res"], st["res"]), st["budget"])
    kw = dict(st["sampler_kw"])
    if st.get("sigma"):
        kw["sigma"] = st["sigma"]
    try:
        sampling.sample(orc, "bisect", st["budget"], **kw)
    except NeedPoints as e:
        return e.pts
    return []


def reconstruct(st):
    ij = np.array(list(st["known_d"].keys()), int)
    v = np.array(list(st["known_d"].values()))
    rec, info = interp.cliff(ij, v, (st["res"], st["res"]), sigma=st.get("sigma", 0.0))
    return rec, info


# ---------------------------------------------------------------- comandos -----
def cmd_init(a):
    st = {"res": a.res, "budget": a.budget, "sigma": a.sigma, "sampler_kw": SAMPLER_KW,
          "known": {}, "rounds": []}
    if a.axes_npz:
        ax = np.load(a.axes_npz)
        # dip.phase_diagram guarda xi ascendente (filas) y ell ascendente; la imagen tiene fila 0 = xi alto
        st["axes"] = {"logxi_rows": np.log10(ax["xi"])[::-1].tolist() if ax["xi"].min() > 0 else ax["xi"][::-1].tolist(),
                      "logell_cols": np.log10(ax["ell"]).tolist() if ax["ell"].min() > 0 else ax["ell"].tolist()}
    json.dump(st, open(a.state, "w"), indent=1)
    print("estado nuevo en", a.state, "(res %d, presupuesto %d, sigma %g)" % (a.res, a.budget, a.sigma))


def cmd_next(a):
    st = load(a.state)
    pts = next_batch(st)
    if not pts:
        print("Presupuesto completo (%d puntos). Siguiente paso: reconstruct." % len(st["known_d"]))
        return
    out = a.out or "tanda_%02d.csv" % (len(st["rounds"]) + 1)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "logxi", "logell", "xi", "ell"])
        for i, j in pts:
            lx, le = to_log(st, i, j)
            w.writerow([i, j, "%.6f" % lx, "%.6f" % le, "%.6g" % 10 ** lx, "%.6g" % 10 ** le])
    st["rounds"].append({"n": len(pts), "file": out})
    save(st, a.state)
    print("Tanda %d: %d puntos -> %s (llevas %d de %d)" % (len(st["rounds"]), len(pts), out,
                                                         len(st["known_d"]), st["budget"]))


def cmd_add(a):
    st = load(a.state)
    n = 0
    for r in csv.DictReader(open(a.results)):
        if "row" in r and r.get("row", "") != "":
            i, j = int(r["row"]), int(r["col"])
        else:
            i, j = to_pix(st, float(r["logxi"]), float(r["logell"]))
        st["known_d"][(i, j)] = float(np.clip(float(r["value"]), 0.0, 1.0))
        n += 1
    save(st, a.state)
    print("cargados %d resultados (total %d de %d)" % (n, len(st["known_d"]), st["budget"]))


def cmd_reconstruct(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = load(a.state)
    rec, info = reconstruct(st)
    np.save(a.out + ".npy", rec)
    lx, le = axes(st)
    ij = np.array(list(st["known_d"].keys()), int)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rec, cmap="viridis", vmin=0, vmax=1, extent=[le[0], le[-1], lx[-1], lx[0]], aspect="auto")
    ax.scatter(le[ij[:, 1]], lx[ij[:, 0]], s=10, c="w", edgecolors="k", linewidths=0.4)
    ax.set_xlabel(r"$\log(\ell)$")
    ax.set_ylabel(r"$\log(\Xi)$")
    ax.set_title("Reconstrucción con %d puntos KMC" % len(ij))
    fig.colorbar(im, label="SoC")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=130)
    print("->", a.out + ".npy", a.out + ".png", "(acantilado %s)" % ("detectado" if info else "NO detectado"))


def cmd_simulate(a):
    """De punta a punta contra un mapa del continuo: next -> 'correr KMC' (oraculo) -> add."""
    from autoexp.oracle import Oracle, load_truth
    from skimage.metrics import peak_signal_noise_ratio

    orc = Oracle(a.g, budget=a.budget, noise=a.sigma)
    st = {"res": 128, "budget": a.budget, "sigma": a.sigma, "sampler_kw": SAMPLER_KW, "known_d": {}, "rounds": []}
    while True:
        pts = next_batch(st)
        if not pts:
            break
        vals = orc.query(pts)
        st["known_d"].update({p: float(x) for p, x in zip(pts, vals)})
        st["rounds"].append(len(pts))
    rec, _ = reconstruct(st)
    ps = peak_signal_noise_ratio(load_truth(a.g), rec, data_range=1.0)
    print("g=%s sigma=%g: %d puntos en %d tandas %s -> PSNR %.2f dB" % (
        a.g, a.sigma, len(st["known_d"]), len(st["rounds"]), st["rounds"], ps))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init")
    p.add_argument("--state", required=True)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--axes-npz")
    p = sp.add_parser("next")
    p.add_argument("--state", required=True)
    p.add_argument("--out")
    p = sp.add_parser("add")
    p.add_argument("--state", required=True)
    p.add_argument("results")
    p = sp.add_parser("reconstruct")
    p.add_argument("--state", required=True)
    p.add_argument("--out", default="mapa_kmc")
    p = sp.add_parser("simulate")
    p.add_argument("--g", default="-2.0")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.0)
    a = ap.parse_args()
    {"init": cmd_init, "next": cmd_next, "add": cmd_add, "reconstruct": cmd_reconstruct,
     "simulate": cmd_simulate}[a.cmd](a)


if __name__ == "__main__":
    main()
