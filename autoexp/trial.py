"""Un intento de reconstruccion con presupuesto de puntos -- MODIFICABLE por el loop.

Uso:
    python -m autoexp.trial <nombre> '<params json>' [--gs=-4.0,-2.0,-0.5] [--note ...]

params (todo opcional salvo lo marcado):
    n            presupuesto de consultas (def 64)
    sampler      grid | halton | uniform | adaptive           (def grid)
    sampler_kw   dict para la estrategia (ver autoexp/sampling.py)
    recon        "dip" o un metodo de autoexp.interp.METHODS  (def rbf_tps)
    aug          null o dict para interp.pseudo_points (solo con recon=dip)
    dip          dict de variables de entorno para dip.restoration
                 (NUM_ITER, LR, REG_NOISE_STD, INPUT_TYPE, NET_WIDTH, NUM_SCALES, ...)

Escribe en autoexp/runs/<nombre>/g<val>/ (queries.json, restored.npy, ...) y corre
autoexp.eval sobre autoexp/runs/<nombre>/. Con recon=dip, las corridas de los g van
en paralelo (misma GPU); el python de DIP se elige con $DIP_PY (def ./venv/bin/python).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

from autoexp import interp, sampling
from autoexp.oracle import Oracle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEV_GS = ["-4.0", "-2.0", "-0.5"]
ALL_GS = ["-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0"]

DIP_DEFAULTS = {
    "NUM_ITER": 8000, "LR": 0.001, "REG_NOISE_STD": 0.08, "PSNR_DROP_TOL": -8,
    "MAX_FALLBACKS": 3, "SHOW_EVERY": 200, "SNAPSHOT_EVERY": 10 ** 9, "SEED": 42,
}


def prepare_g(g, gdir, p):
    """Muestrea y deja listo lo que necesita la reconstruccion. Devuelve el env de DIP o None."""
    os.makedirs(gdir, exist_ok=True)
    orc = Oracle(g, budget=p.get("n", 64))
    ij, v = sampling.sample(orc, p.get("sampler", "grid"), p.get("n", 64), **p.get("sampler_kw", {}))
    orc.save(os.path.join(gdir, "queries.json"))
    H, W = orc.shape
    recon = p.get("recon", "rbf_tps")

    if recon in ("front_split", "cliff"):
        rec, _ = getattr(interp, recon)(ij, v, (H, W), **p.get("recon_kw", {}))
        np.save(os.path.join(gdir, "restored.npy"), rec)
        return None
    if recon != "dip":
        np.save(os.path.join(gdir, "restored.npy"), interp.reconstruct(ij, v, (H, W), recon, **p.get("recon_kw", {})))
        return None

    # DIP: target = valores conocidos en los pixeles de la mascara (el resto no entra
    # en la loss, se rellena con el interpolante solo para que la imagen sea legible).
    mask = np.zeros((H, W), bool)
    mask[ij[:, 0], ij[:, 1]] = True
    aug = p.get("aug")
    if aug:
        pij, pv, target = interp.pseudo_points(ij, v, (H, W), **aug)
        mask[pij[:, 0], pij[:, 1]] = True
    else:
        target = interp.reconstruct(ij, v, (H, W), "nearest")
    target[ij[:, 0], ij[:, 1]] = v  # los reales siempre con su valor exacto
    Image.fromarray(np.round(target * 255).astype(np.uint8), "L").save(os.path.join(gdir, "target.png"))
    np.save(os.path.join(gdir, "mask.npy"), mask)

    env = dict(os.environ)
    env.update({k: str(x) for k, x in DIP_DEFAULTS.items()})
    env.update({k: str(x) for k, x in p.get("dip", {}).items()})
    env.update({
        "IMAGE_PATH": os.path.join(gdir, "target.png"), "MASK_PATH": os.path.join(gdir, "mask.npy"),
        "OUTPUT_DIR": os.path.join(gdir, "dip"), "N_CHANNELS": "1", "PYTHONUNBUFFERED": "1",
    })
    return env


def run(name, p, gs, note="", log=True, runs_dir=None):
    run_dir = os.path.join(runs_dir or os.path.join(HERE, "runs"), name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(p, f, indent=2)

    procs = []
    for g in gs:
        gdir = os.path.join(run_dir, "g%s" % g)
        env = prepare_g(g, gdir, p)
        if env is not None:
            logf = open(os.path.join(gdir, "dip.log"), "w")
            py = os.environ.get("DIP_PY", os.path.join(REPO, "venv", "bin", "python"))
            procs.append((gdir, subprocess.Popen([py, "-m", "dip.restoration"], cwd=REPO, env=env,
                                                 stdout=logf, stderr=subprocess.STDOUT)))
    for gdir, pr in procs:
        if pr.wait() != 0:
            raise RuntimeError("DIP fallo en %s (ver dip.log)" % gdir)
        os.replace(os.path.join(gdir, "dip", "restored.npy"), os.path.join(gdir, "restored.npy"))

    cmd = [sys.executable, "-m", "autoexp.eval", run_dir, "--note", note]
    if not log:
        cmd.append("--no-log")
    subprocess.run(cmd, cwd=REPO, check=True)
    with open(os.path.join(run_dir, "score.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("params", nargs="?", default="{}")
    ap.add_argument("--gs", default=",".join(DEV_GS), help="lista de g, o 'all'")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    gs = ALL_GS if a.gs == "all" else a.gs.split(",")
    run(a.name, json.loads(a.params), gs, a.note)


if __name__ == "__main__":
    main()
