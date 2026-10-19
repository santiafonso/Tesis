"""Control extra en una zona elegida a mano (el control comun, kmc_control pick, solo sortea en
log l >= -2.2 porque mas a la izquierda las corridas no terminaban en 8 h: la zona de log l bajo
quedaba sin controlar). Puntos = grilla log l x log Xi; se descartan los que caen a < 4 px de uno
ya medido. Se corren como un step en el job contenedor (auto_args.reuse_job) con un tope largo,
se traen y se evalua igual que el control comun.

Uso (desde la raiz del repo, en segundo plano; en Mendieta con ~/venv-autoexp/bin/python):
    nohup python -m autoexp.kmc_control_zona --state autoexp/kmc/g3/kmc.json \
        --logell -3.0 -2.5 -2.0 --logxi -1.5 -2.0 --time 30:00:00 >> autoexp/kmc/g3/control_zona.log 2>&1 &
-> autoexp/kmc/g3/control_zona.{csv,json,png}
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

from autoexp import kmc_auto, kmc_planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--logell", type=float, nargs="+", required=True)
    ap.add_argument("--logxi", type=float, nargs="+", required=True)
    ap.add_argument("--time", default="30:00:00", help="tope por corrida (log l = -3 tarda 1.5-3 dias)")
    ap.add_argument("--name", default="control_zona")
    z = ap.parse_args()

    a = argparse.Namespace()  # defaults de kmc_auto + las opciones propias de este g (auto_args)
    for k, v in dict(host="siaosorio@serafin.ccad.unc.edu.ar", fast_host="siaosorio@mulatona.ccad.unc.edu.ar",
                     slow_host="siaosorio@mulatona.ccad.unc.edu.ar", fast_logell=-1.2, remote="~/kmc/tandas",
                     tag="g", poll=600, reuse_job=None, reuse_cores=60, race=False,
                     scratch=os.path.expanduser("~/.cache/kmc_auto")).items():
        setattr(a, k, v)
    st = kmc_planner.load(z.state)
    for k, v in st.get("auto_args", {}).items():
        setattr(a, k, v)
    a.time = z.time
    here = os.path.dirname(z.state)
    lx, le = kmc_planner.axes(st)
    known = np.array(list(st["known_d"].keys()))
    rows = []
    for l_ in z.logell:
        for x_ in z.logxi:
            i, j = int(np.argmin(np.abs(lx - x_))), int(np.argmin(np.abs(le - l_)))
            if np.min(np.hypot(*(known - [i, j]).T)) < 4:
                continue
            rows.append([str(i), str(j), "%.6f" % lx[i], "%.6f" % le[j], "%.6g" % 10 ** lx[i], "%.6g" % 10 ** le[j]]
                        + (["%g" % st["g"]] if "g" in st else []))
    csv_local = os.path.join(here, z.name + ".csv")
    name = "%s_%s" % (a.tag, z.name)
    parts = kmc_auto.launch(a, name, rows, csv_local)
    kmc_auto.log("%s: %d puntos -> %s" % (name, len(rows), ", ".join(p["job"] for p in parts)))
    while not kmc_auto.all_done(parts):
        time.sleep(a.poll)
    local = os.path.join(a.scratch, name)
    kmc_auto.fetch(parts, local)
    res = os.path.join(here, "resultados_%s.csv" % z.name)
    subprocess.run([sys.executable, "-m", "autoexp.kmc_results", local, "--out", res], check=True)
    final = os.path.join(here, "dip", "final.npy")
    subprocess.run([sys.executable, "-m", "autoexp.kmc_control", "eval", "--state", z.state, "--results", res,
                    "--final", final, "--out", os.path.join(here, z.name)], check=True)
    kmc_auto.log("LISTO %s: %s/%s.{json,png}" % (z.name, here, z.name))


if __name__ == "__main__":
    main()
