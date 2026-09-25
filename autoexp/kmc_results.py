"""Lee las salidas del KMC (datos-*.dat) y da, por punto, el SoC final y si la corrida termino.

Formato de datos-*.dat (KMC-Galvanostatic_noclus_param_claude.cpp, GrabaDif): encabezado
`SoC E[V] Tiempo logChi logEle` y una fila por intervalo de tiempo, con E[V] = -mu * 1e-3 (mu =
sobrepotencial promediado en el intervalo). El KMC corre mientras mu < muoff = 149 mV, o sea
hasta que E baja a ~ -0.149 V.

Estado de cada corrida:
  terminado  la ultima E (finita) esta a menos de `tol` del corte: el SoC final es el de esa fila
  cortado    se corto antes (tope de tiempo): el SoC final no se conoce (solo que es >= el ultimo)
  vacio      no llego a escribir ninguna fila
Las filas no finitas (el ultimo intervalo vacio escribe `-inf`) se descartan.

Uso:
    python -m autoexp.kmc_results <carpeta con datos-*.dat> [--out resultados.csv] [--cutoff 0.149]
Escribe un CSV `logxi,logell,value,status,filas,tiempo,archivo` que `kmc_planner add` acepta
(solo toma las filas con status=terminado).
"""
import argparse
import csv
import glob
import os

import numpy as np


def read_run(path, cutoff=0.149, tol=0.02):
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
    if not rows:
        return {"status": "vacio", "filas": 0, "archivo": os.path.basename(path)}
    soc, E, t, lx, le = rows[-1]
    done = E <= -(cutoff - tol)
    return {"logxi": lx, "logell": le, "value": soc, "E_final": E, "tiempo": t, "filas": len(rows),
            "status": "terminado" if done else "cortado", "archivo": os.path.basename(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--out", default="resultados_kmc.csv")
    ap.add_argument("--cutoff", type=float, default=0.149)
    a = ap.parse_args()
    res = [read_run(f, a.cutoff) for f in sorted(glob.glob(os.path.join(a.carpeta, "datos-*.dat")))]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["logxi", "logell", "value", "status", "filas", "tiempo", "archivo"])
        for r in res:
            w.writerow([r.get("logxi", ""), r.get("logell", ""), r.get("value", ""), r["status"], r["filas"],
                        r.get("tiempo", ""), r["archivo"]])
    n = {s: sum(r["status"] == s for r in res) for s in ("terminado", "cortado", "vacio")}
    print("%d corridas: %d terminadas, %d cortadas, %d vacias -> %s" % (len(res), n["terminado"], n["cortado"], n["vacio"], a.out))
    for r in res:
        if r["status"] != "vacio":
            print("  %-9s log Xi %6.2f  log l %6.2f  SoC %.3f  E final %+.3f  (%d filas)" % (
                r["status"], r["logxi"], r["logell"], r["value"], r["E_final"], r["filas"]))


if __name__ == "__main__":
    main()
