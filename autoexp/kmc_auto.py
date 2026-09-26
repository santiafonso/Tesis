"""Loop automatico del KMC real: next -> Serafin (kmc_tanda.slurm) -> esperar -> traer ->
kmc_results -> add --monotone -> figura -> next ... hasta completar el presupuesto, y al final
reconstruct. Se puede cortar y relanzar: el estado (incluido el job en vuelo) vive en kmc.json.

Uso (desde la raiz del repo, en segundo plano):
    nohup ./venv/bin/python -m autoexp.kmc_auto --state autoexp/kmc/g-4/kmc.json > autoexp/kmc/g-4/auto.log 2>&1 &
Env/args: --host (siaosorio@serafin...), --remote (~/kmc/tandas), --tag (g-4), --time (tope del job
por tanda, 1-00:00:00: lo que no termina queda como cota y entra por monotonia en l), --poll (s).
"""
import argparse
import csv
import os
import subprocess
import sys
import time

from autoexp import kmc_planner, kmc_results


def sh(host, cmd, check=True):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, cmd],
                       capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError("ssh %s: %s" % (cmd, r.stderr[-500:]))
    return r.stdout


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def write_tanda(st, pts, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "logxi", "logell", "xi", "ell"])
        for i, j in pts:
            lx, le = kmc_planner.to_log(st, i, j)
            w.writerow([i, j, "%.6f" % lx, "%.6f" % le, "%.6g" % 10 ** lx, "%.6g" % 10 ** le])


def submit(a, st, k, pts):
    here = os.path.dirname(a.state)
    csv_local = os.path.join(here, "tanda_%02d.csv" % k)
    write_tanda(st, pts, csv_local)
    rdir = "%s/%s_t%02d" % (a.remote, a.tag, k)
    sh(a.host, "mkdir -p %s/logs" % rdir)
    subprocess.run(["scp", "-q", csv_local, "%s:%s/tanda.csv" % (a.host, rdir)], check=True)
    out = sh(a.host, "cd %s && sbatch --parsable --time=%s ~/kmc/kmc_tanda.slurm tanda.csv" % (rdir, a.time))
    job = out.strip().splitlines()[-1].split(";")[0]
    st.setdefault("jobs", {})[str(k)] = {"job": job, "remote": rdir, "n": len(pts), "sent": time.time()}
    st["pending"] = [list(p) for p in pts]
    if len(st["rounds"]) < k:
        st["rounds"].append({"n": len(pts), "file": os.path.basename(csv_local)})
    kmc_planner.save(st, a.state)
    log("tanda %d: %d puntos, job %s -> %s" % (k, len(pts), job, rdir))


def collect(a, st, k):
    here = os.path.dirname(a.state)
    info = st["jobs"][str(k)]
    local = os.path.join(a.scratch, "%s_t%02d" % (a.tag, k))
    os.makedirs(local, exist_ok=True)
    subprocess.run(["rsync", "-aq", "--exclude", "vmd-*", "--exclude", "*.xyz",
                    "%s:%s/" % (a.host, info["remote"]), local + "/"], check=True)
    res = os.path.join(here, "resultados_t%02d.csv" % k)
    subprocess.run([sys.executable, "-m", "autoexp.kmc_results", local, "--out", res,
                    "--runs", os.path.join(here, "corridas_t%02d.csv" % k)], check=True)
    subprocess.run([sys.executable, "-m", "autoexp.kmc_planner", "add", "--state", a.state, "--monotone", res],
                   check=True)
    st = kmc_planner.load(a.state)
    st["jobs"][str(k)]["done"] = time.time()
    st["pending"] = []
    kmc_planner.save(st, a.state)
    plot(a)
    log("tanda %d cargada: %d de %d puntos" % (k, len(st["known_d"]), st["budget"]))


def plot(a):
    subprocess.run([sys.executable, "-m", "autoexp.kmc_plot", "--state", a.state], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--host", default="siaosorio@serafin.ccad.unc.edu.ar")
    ap.add_argument("--remote", default="~/kmc/tandas")
    ap.add_argument("--tag", default="g-4")
    ap.add_argument("--time", default="1-00:00:00")
    ap.add_argument("--poll", type=int, default=600)
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/kmc_auto"))
    a = ap.parse_args()

    while True:
        st = kmc_planner.load(a.state)
        open_jobs = {int(k): v for k, v in st.get("jobs", {}).items() if "done" not in v}
        if open_jobs:
            k, info = min(open_jobs.items())
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", a.host,
                                "sacct -n -X -P -j %s -o State" % info["job"]], capture_output=True, text=True)
            q = r.stdout.strip().split()[0] if r.returncode == 0 and r.stdout.strip() else "?"  # ? = red caida
            if q not in ("?", "PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING"):
                collect(a, st, k)
                continue
            time.sleep(a.poll)
            continue
        pts = kmc_planner.next_batch(st)
        if not pts:
            here = os.path.dirname(a.state)
            subprocess.run([sys.executable, "-m", "autoexp.kmc_planner", "reconstruct", "--state", a.state,
                            "--out", os.path.join(here, "mapa_kmc")], check=True)
            plot(a)
            log("presupuesto completo: %d puntos. Mapa en %s/mapa_kmc.{npy,png}" % (len(st["known_d"]), here))
            return
        k = max([len(st["rounds"])] + [int(x) for x in st.get("jobs", {})]) + 1
        submit(a, st, k, pts)


if __name__ == "__main__":
    main()
