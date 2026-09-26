"""Lee una tanda de KMC corrida con slurm/kmc/kmc_tanda.slurm y da, por punto, el SoC final
promediado sobre las replicas que terminaron.

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
RUN_RE = re.compile(r"r(\d+)_c(\d+)_k(\d+)$")


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
    r = {"run": os.path.basename(d), "row": int(m.group(1)), "col": int(m.group(2)), "rep": int(m.group(3)),
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


def aggregate(runs):
    by = defaultdict(list)
    for r in runs:
        by[(r["row"], r["col"])].append(r)
    pts = []
    for (i, j), rs in sorted(by.items()):
        ok = [r["value"] for r in rs if r["status"] == "terminado"]
        lb = [r["value"] for r in rs if r["status"] in ("cortado", "en_curso") and r["filas"]]
        pts.append({"row": i, "col": j, "logxi": next((r["logxi"] for r in rs if r["logxi"] != ""), ""),
                    "logell": next((r["logell"] for r in rs if r["logell"] != ""), ""),
                    "value": float(np.mean(ok)) if ok else (max(lb) if lb else ""), "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else "",
                    "n_term": len(ok), "n_runs": len(rs), "status": "terminado" if ok else "cota" if lb else
                    "/".join(sorted({r["status"] for r in rs}))})
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tanda", help="carpeta con runs/r*_c*_k*/")
    ap.add_argument("--out", default="resultados_kmc.csv")
    ap.add_argument("--runs", help="CSV opcional con una fila por corrida")
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.tanda, "runs", "r*_c*_k*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit("no hay corridas en %s/runs/" % a.tanda)
    runs = [read_run(d) for d in dirs]
    pts = aggregate(runs)

    cols = ["row", "col", "logxi", "logell", "value", "std", "n_term", "n_runs", "status"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(pts)
    if a.runs:
        rcols = ["run", "row", "col", "rep", "logxi", "logell", "value", "E_final", "filas", "rc", "segundos", "status", "nota"]
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
        print("  r%3d c%3d  log Xi %s  log l %s  %s%-9s  %d/%d  %s" % (
            p["row"], p["col"], lx, le, v, s, p["n_term"], p["n_runs"], p["status"]))
    stds = [p["std"] for p in pts if p["std"] != ""]
    if stds:
        print("sigma entre replicas: %.4f (raiz del promedio de varianzas, %d puntos)" % (
            np.sqrt(np.mean(np.square(stds))), len(stds)))


if __name__ == "__main__":
    main()
