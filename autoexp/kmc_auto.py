"""Loop automatico del KMC real: next -> cluster(s) (kmc_tanda.slurm) -> esperar -> traer ->
kmc_results -> add --monotone -> figura -> next ... hasta completar el presupuesto; al final
reconstruccion TPS + DIP (Mendieta) y tanda de control. Se puede cortar y relanzar: el estado
(incluidos los jobs en vuelo) vive en kmc.json.

Cada tanda se parte por costo: el tiempo de una corrida va como ~1/l (medido: log l -0.5 ~6 min,
-1.5 ~45 min, -2.5 ~6.5 h). Los puntos rapidos (log l > --fast-logell) van a Mulatona con pocos
nucleos (particion short, 1 h: entran facil en la cola); los lentos a Serafin, nodo entero, tope
--time (lo que no termina queda como cota y entra por monotonia en l).

Uso (desde la raiz del repo, en segundo plano):
    nohup ./venv/bin/python -m autoexp.kmc_auto --state autoexp/kmc/g-4/kmc.json >> autoexp/kmc/g-4/auto.log 2>&1 &
"""
import argparse
import csv
import os
import subprocess
import sys
import time

from autoexp import kmc_planner

ACTIVE = ("?", "PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING")


def sh(host, cmd, check=True):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, cmd],
                       capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError("ssh %s: %s" % (cmd, r.stderr[-500:]))
    return r.stdout


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def job_state(host, job):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host,
                        "sacct -n -X -P -j %s -o State" % job], capture_output=True, text=True)
    return r.stdout.strip().split()[0] if r.returncode == 0 and r.stdout.strip() else "?"  # ? = red caida


def write_rows(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "logxi", "logell", "xi", "ell"])
        w.writerows(rows)


def rows_for(st, pts):
    out = []
    for i, j in pts:
        lx, le = kmc_planner.to_log(st, i, j)
        out.append([i, j, "%.6f" % lx, "%.6f" % le, "%.6g" % 10 ** lx, "%.6g" % 10 ** le])
    return out


def launch(a, name, rows, local_csv):
    """Parte las filas por costo y manda cada parte a su cluster. Devuelve la lista de partes."""
    write_rows(rows, local_csv)
    fast = [r for r in rows if float(r[3]) > a.fast_logell]
    slow = [r for r in rows if float(r[3]) <= a.fast_logell]
    parts = []
    for host, sub, opts in ((a.fast_host, fast, "-p short --time=00:59:00 -c %d" % min(32, max(2, 2 * len(fast)))),
                            (a.host, slow, "--time=%s" % a.time)):
        if not sub:
            continue
        rdir = "%s/%s" % (a.remote, name)
        tmp = local_csv + ".part"
        write_rows(sub, tmp)
        sh(host, "mkdir -p %s/logs" % rdir)
        subprocess.run(["scp", "-q", tmp, "%s:%s/tanda.csv" % (host, rdir)], check=True)
        job = sh(host, "cd %s && sbatch --parsable %s ~/kmc/kmc_tanda.slurm tanda.csv" % (rdir, opts)
                 ).strip().splitlines()[-1].split(";")[0]
        os.remove(tmp)
        parts.append({"host": host, "job": job, "remote": rdir, "n": len(sub)})
        log("%s: %d puntos -> %s job %s" % (name, len(sub), host.split("@")[1].split(".")[0], job))
    return parts


def parts_of(info, a):
    return info.get("parts") or [{"host": a.host, "job": info["job"], "remote": info["remote"]}]  # formato viejo


def all_done(parts):
    return all(job_state(p["host"], p["job"]) not in ACTIVE for p in parts)


def fetch(parts, local):
    os.makedirs(local, exist_ok=True)
    for p in parts:
        subprocess.run(["rsync", "-aq", "--exclude", "vmd-*", "--exclude", "*.xyz",
                        "%s:%s/" % (p["host"], p["remote"]), local + "/"], check=True)


def submit(a, st, k, pts):
    here = os.path.dirname(a.state)
    csv_local = os.path.join(here, "tanda_%02d.csv" % k)
    parts = launch(a, "%s_t%02d" % (a.tag, k), rows_for(st, pts), csv_local)
    st.setdefault("jobs", {})[str(k)] = {"parts": parts, "n": len(pts), "sent": time.time()}
    st["pending"] = [list(p) for p in pts]
    if len(st["rounds"]) < k:
        st["rounds"].append({"n": len(pts), "file": os.path.basename(csv_local)})
    kmc_planner.save(st, a.state)


def collect(a, st, k):
    here = os.path.dirname(a.state)
    local = os.path.join(a.scratch, "%s_t%02d" % (a.tag, k))
    fetch(parts_of(st["jobs"][str(k)], a), local)
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


def final_dip(a, here):
    """Reconstruccion final TPS + DIP (autoexp.kmc_dip) con DIP en Mendieta."""
    out = os.path.join(here, "dip")
    rel = os.path.relpath(out, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "-m", "autoexp.kmc_dip", "prepare", "--state", a.state, "--out", out], check=True)
    sh(a.dip_host, "mkdir -p ~/Tesis-autoexp/%s ~/Tesis-autoexp/slurm" % rel)
    subprocess.run(["rsync", "-aq", out + "/", "%s:Tesis-autoexp/%s/" % (a.dip_host, rel)], check=True)
    subprocess.run(["scp", "-q", "slurm/kmc_dip.slurm", "%s:Tesis-autoexp/slurm/" % a.dip_host], check=True)
    job = sh(a.dip_host, "cd ~/Tesis-autoexp && sbatch --parsable --export=ALL,DIPDIR=%s slurm/kmc_dip.slurm"
             % rel).strip().split(";")[0]
    log("DIP final: job %s en Mendieta" % job)
    while True:
        time.sleep(120)
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", a.dip_host, "sacct -n -X -P -j %s -o State" % job],
                           capture_output=True, text=True)
        q = r.stdout.strip().split()[0] if r.returncode == 0 and r.stdout.strip() else "?"
        if q not in ("?", "PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
            break
    if q != "COMPLETED":
        log("DIP final termino con %s: queda solo la TPS (mapa_kmc)" % q)
        return
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=restored.npy", "--include=dip.log", "--exclude=*",
                    "%s:Tesis-autoexp/%s/" % (a.dip_host, rel), out + "/"], check=True)
    subprocess.run([sys.executable, "-m", "autoexp.kmc_dip", "fuse", "--state", a.state, "--out", out], check=True)
    log("LISTO: mapa final TPS + DIP en %s/final.{npy,png}" % out)


def control(a, here):
    """Tanda de control: 10 puntos no usados (kmc_control pick), KMC, error contra el mapa final."""
    final = os.path.join(here, "dip", "final.npy")
    if not os.path.exists(final):
        final = os.path.join(here, "mapa_kmc.npy")
    csv_local = os.path.join(here, "control.csv")
    if not os.path.exists(csv_local):
        subprocess.run([sys.executable, "-m", "autoexp.kmc_control", "pick", "--state", a.state, "--out", csv_local],
                       check=True)
    st = kmc_planner.load(a.state)
    if not st.get("control_parts"):
        rows = list(csv.reader(open(csv_local)))[1:]
        st["control_parts"] = launch(a, "%s_control" % a.tag, rows, csv_local)
        kmc_planner.save(st, a.state)
    while not all_done(st["control_parts"]):
        time.sleep(a.poll)
    local = os.path.join(a.scratch, "%s_control" % a.tag)
    fetch(st["control_parts"], local)
    res = os.path.join(here, "resultados_control.csv")
    subprocess.run([sys.executable, "-m", "autoexp.kmc_results", local, "--out", res], check=True)
    subprocess.run([sys.executable, "-m", "autoexp.kmc_control", "eval", "--state", a.state, "--results", res,
                    "--final", final, "--out", os.path.join(here, "control")], check=True)
    log("LISTO control: %s/control.{json,png}" % here)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--host", default="siaosorio@serafin.ccad.unc.edu.ar", help="puntos lentos")
    ap.add_argument("--fast-host", default="siaosorio@mulatona.ccad.unc.edu.ar", help="puntos rapidos")
    ap.add_argument("--fast-logell", type=float, default=-1.2, help="log l > esto -> fast-host")
    ap.add_argument("--remote", default="~/kmc/tandas")
    ap.add_argument("--tag", default="g-4")
    ap.add_argument("--time", default="08:00:00", help="tope de las tandas lentas (Serafin)")
    ap.add_argument("--poll", type=int, default=600)
    ap.add_argument("--dip-host", default="siaosorio@mendieta.ccad.unc.edu.ar")
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/kmc_auto"))
    a = ap.parse_args()

    while True:
        st = kmc_planner.load(a.state)
        open_jobs = {int(k): v for k, v in st.get("jobs", {}).items() if "done" not in v}
        if open_jobs:
            k, info = min(open_jobs.items())
            if all_done(parts_of(info, a)):
                collect(a, st, k)
            else:
                time.sleep(a.poll)
            continue
        pts = kmc_planner.next_batch(st)
        if not pts:
            here = os.path.dirname(a.state)
            subprocess.run([sys.executable, "-m", "autoexp.kmc_planner", "reconstruct", "--state", a.state,
                            "--out", os.path.join(here, "mapa_kmc")], check=True)
            plot(a)
            log("presupuesto completo: %d puntos. Mapa TPS en %s/mapa_kmc.{npy,png}" % (len(st["known_d"]), here))
            final_dip(a, here)
            control(a, here)
            return
        k = max([len(st["rounds"])] + [int(x) for x in st.get("jobs", {})]) + 1
        submit(a, st, k, pts)


if __name__ == "__main__":
    main()
