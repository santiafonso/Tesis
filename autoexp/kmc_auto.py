"""Loop automatico del KMC real: next -> cluster(s) (kmc_tanda.slurm) -> esperar -> traer ->
kmc_results -> add --monotone -> figura -> next ... hasta completar el presupuesto; al final
reconstruccion TPS + DIP (Mendieta) y tanda de control. Se puede cortar y relanzar: el estado
(incluidos los jobs en vuelo) vive en kmc.json.

Cada tanda se parte por costo: el tiempo de una corrida va como ~1/l (medido: log l -0.5 ~6 min,
-1.5 ~45 min, -2.5 ~6.5 h). Los puntos rapidos (log l > --fast-logell) van a Mulatona short (1 h);
los lentos a --slow-host (def. Mulatona mono, 2 replicas por punto, pocos nucleos: entra en la cola
mucho antes que un nodo entero de Serafin), tope --time (lo que no termina queda como cota y entra
por monotonia en l). --host (Serafin) queda para --reuse-job.

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


NET_FAIL = (255, 12, 30, 35)  # ssh/scp: 255; rsync: 12/30/35 (corte de red o timeout)


def sh(host, cmd, check=True, tries=6):
    for t in range(tries):
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, cmd],
                           capture_output=True, text=True)
        if r.returncode != 255 or t == tries - 1:
            break
        time.sleep(30 * (t + 1))  # "Connection reset" del login: esperar y reintentar
    if check and r.returncode:
        raise RuntimeError("ssh %s: %s" % (cmd, r.stderr[-500:]))
    return r.stdout


def net(args, tries=6):
    """scp/rsync con reintentos ante cortes de red."""
    for t in range(tries):
        r = subprocess.run(args)
        if r.returncode == 0:
            return
        if r.returncode not in NET_FAIL or t == tries - 1:
            raise subprocess.CalledProcessError(r.returncode, args)
        time.sleep(30 * (t + 1))


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def job_state(host, job):
    """Estado SLURM de un job, o de un step ("<job>.<n>": tanda lanzada con srun --overlap dentro
    de un job propio que ya tenia nodo)."""
    step = "." in str(job)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host,
                        "sacct -n -P -j %s -o JobID,State%s" % (job, "" if step else " -X")],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        return "?"  # red caida
    for line in r.stdout.strip().splitlines():
        jid, state = line.split("|")[:2]
        if jid == str(job) or not step:
            return state.split()[0]
    return "?"


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


def to_secs(t):
    d, _, t = t.rpartition("-")
    x = 0
    for p in t.split(":"):
        x = x * 60 + int(p)
    return x + int(d or 0) * 86400


def reuse_step(a, name, local_csv):
    """Si --reuse-job es un job propio corriendo con tiempo de sobra (> tope + 30 min), corre la
    tanda entera como step (srun --overlap) en su nodo: sin cola. Devuelve la parte o None."""
    if not a.reuse_job:
        return None
    left = sh(a.host, "squeue -h -j %s -o %%L" % a.reuse_job, check=False).strip()
    tope = to_secs(a.time)
    if not left or not left[0].isdigit() or to_secs(left) < tope + 1800:
        return None
    rdir = "%s/%s" % (a.remote, name)
    sh(a.host, "mkdir -p %s/logs" % rdir)
    net(["scp", "-q", local_csv, "%s:%s/tanda.csv" % (a.host, rdir)])
    before = set(sh(a.host, "squeue -s -h -j %s -o %%i" % a.reuse_job).split())
    # ssh no vuelve mientras srun vive aunque este en segundo plano: se lanza sin esperar y se
    # suelta cuando aparece el step (srun sigue en el login con setsid). -c: nucleos que se usan,
    # dejando lugar a lo que el job ya tenga corriendo.
    pr = subprocess.Popen(["ssh", "-o", "BatchMode=yes", a.host,
                           'cd %s && nohup setsid srun --jobid=%s --overlap -N1 -n1 -c%d --export=ALL,TOPE_MAX=%d '
                           'bash ~/kmc/kmc_tanda.slurm tanda.csv > logs/step.out 2> logs/step.err < /dev/null &'
                           % (rdir, a.reuse_job, a.reuse_cores, tope)],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(12):
        time.sleep(5)
        new = set(sh(a.host, "squeue -s -h -j %s -o %%i" % a.reuse_job).split()) - before
        if new:
            time.sleep(5)
            pr.kill()
            step = sorted(new, key=lambda x: int(x.split(".")[1]) if x.split(".")[1].isdigit() else -1)[-1]
            log("%s: %d puntos -> step %s (nodo del job %s, sin cola)" % (name, len(open(local_csv).readlines()) - 1,
                                                                          step, a.reuse_job))
            return {"host": a.host, "job": step, "remote": rdir}
    pr.kill()
    raise RuntimeError("no aparecio el step en el job %s" % a.reuse_job)


def launch(a, name, rows, local_csv):
    """Parte las filas por costo y manda cada parte a su cluster (o todo a un job propio con
    tiempo, ver reuse_step). Devuelve la lista de partes."""
    write_rows(rows, local_csv)
    part = reuse_step(a, name, local_csv)
    if part:
        return [dict(part, n=len(rows))]
    fast = [r for r in rows if float(r[3]) > a.fast_logell]
    slow = [r for r in rows if float(r[3]) <= a.fast_logell]
    parts = []
    slow_opts = ("-p mono --time=%s -c %d --export=ALL,NREP=2" % (a.time, min(32, 2 * len(slow)))
                 if "mulatona" in a.slow_host else "--time=%s" % a.time)
    for host, sub, opts, sfx in ((a.fast_host, fast, "-p short --time=00:59:00 -c %d" % min(32, max(2, 2 * len(fast))), ""),
                                 (a.slow_host, slow, slow_opts, "_lentos")):
        if not sub:
            continue
        rdir = "%s/%s%s" % (a.remote, name, sfx)
        tmp = local_csv + ".part"
        write_rows(sub, tmp)
        sh(host, "mkdir -p %s/logs" % rdir)
        net(["scp", "-q", tmp, "%s:%s/tanda.csv" % (host, rdir)])
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
        net(["rsync", "-aq", "--exclude", "vmd-*", "--exclude", "*.xyz",
             "%s:%s/" % (p["host"], p["remote"]), local + "/"])


def launch_alt(a, name, local_csv):
    """Copia de la tanda entera en --host (Serafin, nodo entero) que compite con las partes de
    Mulatona: la que arranque primero se queda (resolve_race)."""
    rdir = "%s/%s_alt" % (a.remote, name)
    sh(a.host, "mkdir -p %s/logs" % rdir)
    net(["scp", "-q", local_csv, "%s:%s/tanda.csv" % (a.host, rdir)])
    job = sh(a.host, "cd %s && sbatch --parsable --time=%s ~/kmc/kmc_tanda.slurm tanda.csv" % (rdir, a.time)
             ).strip().splitlines()[-1].split(";")[0]
    log("%s: copia entera -> serafin job %s (compite)" % (name, job))
    return [{"host": a.host, "job": job, "remote": rdir}]


def resolve_race(a, st, k):
    """Si hay copia alternativa (Serafin), gana la que TERMINA primero y la otra se cancela. Se
    decide al terminar y no al arrancar porque Mulatona desaloja (PreemptMode=REQUEUE: la particion
    batch, de otra cuenta, tiene mas prioridad) y un job arrancado puede volver a la cola."""
    info = st["jobs"][str(k)]
    alt = info.get("alt_parts")
    if not alt:
        return
    main_done, alt_done = all_done(info["parts"]), all_done(alt)
    if not (main_done or alt_done):
        return
    lose, win = (alt, info["parts"]) if main_done else (info["parts"], alt)
    for p in lose:
        sh(p["host"], "scancel %s" % p["job"], check=False)
    info["parts"], info["race"] = win, "gano %s" % win[0]["host"].split("@")[1].split(".")[0]
    info.pop("alt_parts")
    kmc_planner.save(st, a.state)
    log("tanda %d: %s; cancelado %s" % (k, info["race"], ",".join(p["job"] for p in lose)))


def submit(a, st, k, pts):
    here = os.path.dirname(a.state)
    csv_local = os.path.join(here, "tanda_%02d.csv" % k)
    parts = launch(a, "%s_t%02d" % (a.tag, k), rows_for(st, pts), csv_local)
    # guardar YA: si despues se corta la red, la tanda queda registrada (antes quedaban jobs huerfanos)
    st.setdefault("jobs", {})[str(k)] = {"parts": parts, "n": len(pts), "sent": time.time()}
    st["pending"] = [list(p) for p in pts]
    if len(st["rounds"]) < k:
        st["rounds"].append({"n": len(pts), "file": os.path.basename(csv_local)})
    kmc_planner.save(st, a.state)
    if a.race and not any(a.host == p["host"] for p in parts):
        try:
            st["jobs"][str(k)]["alt_parts"] = launch_alt(a, "%s_t%02d" % (a.tag, k), csv_local)
            kmc_planner.save(st, a.state)
        except Exception as e:  # la copia es opcional
            log("copia en Serafin no enviada (%s); sigue solo con Mulatona" % str(e)[-200:])


def collect(a, st, k):
    here = os.path.dirname(a.state)
    for p in st["jobs"][str(k)].pop("alt_parts", []):  # copia que sobro (la carrera no se resolvio por un corte de red)
        sh(p["host"], "scancel %s" % p["job"], check=False)
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
    subprocess.run([sys.executable, "-m", "autoexp.kmc_plot_tandas", "--state", a.state], check=False)


def final_dip(a, here):
    """Reconstruccion final TPS + DIP (autoexp.kmc_dip) con DIP en Mendieta."""
    out = os.path.join(here, "dip")
    rel = os.path.relpath(out, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "-m", "autoexp.kmc_dip", "prepare", "--state", a.state, "--out", out], check=True)
    sh(a.dip_host, "mkdir -p ~/Tesis-autoexp/%s ~/Tesis-autoexp/slurm" % rel)
    net(["rsync", "-aq", out + "/", "%s:Tesis-autoexp/%s/" % (a.dip_host, rel)])
    net(["scp", "-q", "slurm/kmc_dip.slurm", "%s:Tesis-autoexp/slurm/" % a.dip_host])
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
    net(["rsync", "-aq", "--include=*/", "--include=restored.npy", "--include=dip.log", "--exclude=*",
                    "%s:Tesis-autoexp/%s/" % (a.dip_host, rel), out + "/"])
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
    ap.add_argument("--slow-host", default="siaosorio@mulatona.ccad.unc.edu.ar",
                    help="puntos lentos: Mulatona mono con pocos nucleos (la cola de nodo entero en Serafin era de ~1.5 dias)")
    ap.add_argument("--fast-logell", type=float, default=-1.2, help="log l > esto -> fast-host")
    ap.add_argument("--remote", default="~/kmc/tandas")
    ap.add_argument("--tag", default="g-4")
    ap.add_argument("--time", default="08:00:00", help="tope de las tandas lentas (Serafin)")
    ap.add_argument("--poll", type=int, default=600)
    ap.add_argument("--reuse-job", help="job propio corriendo en --host cuyo nodo se reusa con srun --overlap")
    ap.add_argument("--reuse-cores", type=int, default=60)
    ap.add_argument("--no-race", dest="race", action="store_false",
                    help="no mandar la copia de la tanda a Serafin que compite con Mulatona")
    ap.add_argument("--dip-host", default="siaosorio@mendieta.ccad.unc.edu.ar")
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/kmc_auto"))
    a = ap.parse_args()

    while True:
        try:
            if step(a):
                return
        except Exception:
            import traceback
            log("ERROR (se reintenta en %d s):\n%s" % (a.poll, traceback.format_exc()[-1500:]))
            time.sleep(a.poll)


def step(a):
    """Una vuelta del loop. Devuelve True cuando termino todo."""
    st = kmc_planner.load(a.state)
    open_jobs = {int(k): v for k, v in st.get("jobs", {}).items() if "done" not in v}
    if open_jobs:
        k, info = min(open_jobs.items())
        resolve_race(a, st, k)
        info = st["jobs"][str(k)]
        if all_done(parts_of(info, a)):
            collect(a, st, k)
        else:
            time.sleep(a.poll)
        return False
    pts = kmc_planner.next_batch(st)
    if not pts:
        here = os.path.dirname(a.state)
        subprocess.run([sys.executable, "-m", "autoexp.kmc_planner", "reconstruct", "--state", a.state,
                        "--out", os.path.join(here, "mapa_kmc")], check=True)
        plot(a)
        log("presupuesto completo: %d puntos. Mapa TPS en %s/mapa_kmc.{npy,png}" % (len(st["known_d"]), here))
        if not os.path.exists(os.path.join(here, "dip", "final.npy")):  # al relanzar, no repetir DIP
            final_dip(a, here)
        control(a, here)
        return True
    k = max([len(st["rounds"])] + [int(x) for x in st.get("jobs", {})]) + 1
    submit(a, st, k, pts)
    return False


if __name__ == "__main__":
    main()
