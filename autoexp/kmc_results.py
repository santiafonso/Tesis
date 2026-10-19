"""Lee una tanda de KMC corrida con slurm/kmc/kmc_tanda.slurm y da, por punto, el SoC final
de las replicas que terminaron (mediana: con replicas bimodales -unas intercalan y otras
se quedan en ~0- el promedio da un valor que ninguna corrida dio; esas quedan con bimodal=1).

Estructura de la tanda: runs/r<row>_c<col>_k<rep>/ con datos-*.dat (salida del KMC:
`SoC E[V] Tiempo logChi logEle`, una fila por intervalo) y estado.txt (lo escribe el slurm):
  rc=0          terminado: el SoC final es el de la ultima fila finita
  rc=124        cortado por el tope de tiempo: el SoC real es >= el ultimo (no se usa)
  rc=corriendo  en curso
  rc=<otro>     error (ver kmc.out en la carpeta)
Con rc=0 y ninguna fila, el sobrepotencial paso el corte en el primer intervalo: SoC = 0 (pasa en
toda la zona de Xi bajo, donde el continuo tambien da 0). Si termino con filas pero el SoC nunca
supero `SOC_MIN`, se marca `sospechoso` y no se carga (visto en log Xi 1.43, log l -3.53: el KMC
recorre todos los intervalos sin intercalar, y el continuo da 0.996 ahi).
El estado sale del codigo de salida, no de la ultima E: el ultimo intervalo no siempre se
imprime (y cuando se imprime vacio da `-inf`, que se descarta).

Uso:
    python -m autoexp.kmc_results <carpeta de la tanda> [--out resultados.csv] [--runs corridas.csv]
Escribe `row,col,logxi,logell,value,std,n_term,n_runs,status` por punto (status=terminado si
termino al menos una replica), que `kmc_planner add` acepta tal cual. Si ninguna termino pero
alguna cortada / en curso escribio filas, status=cota y value = el mayor SoC alcanzado (el SoC
real es >= eso); `kmc_planner add --monotone` lo usa como pseudo-punto. La dispersion entre
replicas estima el `sigma` del planificador.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

SOC_MIN = 0.01
RUN_RE = re.compile(r"(?:g([-\d.]+)_)?r(\d+)_c(\d+)_k(\d+)$")


def read_datos(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        try:
            v = [float(x) for x in p[:5]]
        except ValueError:
            continue  # encabezado
        if all(np.isfinite(v)):
            rows.append(v)
    # Algunas corridas siguen escribiendo despues del final: un bloque de filas con SoC ~0 (0, -0.008, -0.22,
    # 0.010...) y E 0, 0.001 o +-inf, con t que salta. Las no finitas ya se descartaron; las otras se tomaban por
    # la ultima (SoC ~0 donde el real es 0.83-0.97). El SoC no baja durante la descarga: se saca la caida final.
    top = max((r[0] for r in rows), default=0.0)
    if top > SOC_MIN:
        while rows and rows[-1][0] < 0.5 * top:
            rows.pop()
    return rows


def read_estado(path):
    if not os.path.exists(path):
        return {}
    return dict(kv.split("=", 1) for kv in open(path).read().split() if "=" in kv)


def status_of(rc):
    return {None: "sin_estado", "corriendo": "en_curso", "0": "terminado", "124": "cortado"}.get(rc, "error")


def read_run(d):
    m = RUN_RE.search(os.path.basename(d.rstrip("/")))
    est = read_estado(os.path.join(d, "estado.txt"))
    datos = sorted(glob.glob(os.path.join(d, "datos-*.dat")))
    rows = read_datos(datos[0]) if datos else []
    r = {"run": os.path.basename(d), "g": m.group(1) or est.get("g", "-4"),
         "row": int(m.group(2)), "col": int(m.group(3)), "rep": int(m.group(4)),
         "rc": est.get("rc", ""), "segundos": est.get("segundos", ""), "filas": len(rows)}
    r["status"] = status_of(est.get("rc"))
    if r["status"] == "terminado" and not rows:
        r.update(value=0.0, E_final="", nota="corte en el primer intervalo")
    elif r["status"] == "terminado" and max(x[0] for x in rows) < SOC_MIN and len(rows) > 5:
        r["status"] = "sospechoso"
    if rows:
        soc, E, t, lx, le = rows[-1]
        r.update(value=soc, E_final=E, logxi=lx, logell=le)
    else:
        r.update(logxi=float(est["xi"]) if "xi" in est else "", logell=float(est["el"]) if "el" in est else "")
    return r


def bimodal(ok, hi=0.5, lo=0.05):
    """Replicas partidas: unas intercalan (>= hi) y otras se quedan en ~0 (g=-8, log Xi alto: 0.99/0.99/0.99/0.00).
    El promedio da un valor que ninguna corrida dio; por eso el valor del punto es la mediana."""
    return len(ok) > 1 and max(ok) >= hi and min(ok) < lo


def aggregate(runs):
    by = defaultdict(list)
    for r in runs:
        by[(float(r["g"]), r["row"], r["col"])].append(r)
    pts = []
    for (g, i, j), rs in sorted(by.items()):
        ok = [r["value"] for r in rs if r["status"] == "terminado"]
        lb = [r["value"] for r in rs if r["status"] in ("cortado", "en_curso") and r["filas"]]
        # Replicas terminadas por debajo de lo que ya alcanzaron las cortadas (zona bimodal de log Xi alto):
        # el valor terminado no representa el punto -> cota con el mayor SoC visto.
        bajo_cota = bool(ok and lb and max(lb) > float(np.median(ok)) + 0.05)
        if bajo_cota:
            lb, ok = lb + ok, []
        pts.append({"g": g, "row": i, "col": j, "logxi": next((r["logxi"] for r in rs if r["logxi"] != ""), ""),
                    "logell": next((r["logell"] for r in rs if r["logell"] != ""), ""),
                    "value": float(np.median(ok)) if ok else (max(lb) if lb else ""), "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else "",
                    "n_term": len(ok), "n_runs": len(rs), "bimodal": int(bimodal(ok)), "status": "terminado" if ok else "cota" if lb else
                    "/".join(sorted({r["status"] for r in rs}))})
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tanda", help="carpeta con runs/r*_c*_k*/")
    ap.add_argument("--out", default="resultados_kmc.csv")
    ap.add_argument("--runs", help="CSV opcional con una fila por corrida")
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.tanda, "runs", "*r*_c*_k*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit("no hay corridas en %s/runs/" % a.tanda)
    runs = [read_run(d) for d in dirs]
    pts = aggregate(runs)

    cols = ["g", "row", "col", "logxi", "logell", "value", "std", "n_term", "n_runs", "status", "bimodal"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(pts)
    if a.runs:
        rcols = ["run", "g", "row", "col", "rep", "logxi", "logell", "value", "E_final", "filas", "rc", "segundos", "status", "nota"]
        with open(a.runs, "w", newline="") as f:
            w = csv.DictWriter(f, rcols, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(runs)

    cnt = defaultdict(int)
    for r in runs:
        cnt[r["status"]] += 1
    print("%d corridas: %s" % (len(runs), ", ".join("%d %s" % (n, s) for s, n in sorted(cnt.items()))))
    nt = sum(p["status"] == "terminado" for p in pts)
    print("%d puntos, %d con al menos una replica terminada -> %s" % (len(pts), nt, a.out))
    for p in pts:
        v = "SoC %.3f" % p["value"] if p["value"] != "" else "         "
        s = " +- %.3f" % p["std"] if p["std"] != "" else ""
        lx = "%6.2f" % p["logxi"] if p["logxi"] != "" else "     ?"
        le = "%6.2f" % p["logell"] if p["logell"] != "" else "     ?"
        print("  g%5.1f r%3d c%3d  log Xi %s  log l %s  %s%-9s  %d/%d  %s" % (
            p["g"], p["row"], p["col"], lx, le, v, s, p["n_term"], p["n_runs"], p["status"]))
    stds = [p["std"] for p in pts if p["std"] != ""]
    if stds:
        print("sigma entre replicas: %.4f (raiz del promedio de varianzas, %d puntos)" % (
            np.sqrt(np.mean(np.square(stds))), len(stds)))


if __name__ == "__main__":
    main()
