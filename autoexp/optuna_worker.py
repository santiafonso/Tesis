"""Worker de Optuna (puntos 1-3 de la reunion del 17/9) -- MODIFICABLE por el loop.

Busca a la vez muestreo (la grilla como un hiperparametro mas, o adaptativo),
pseudo-puntos de interpolacion, e hiperparametros de DIP, con presupuesto fijo de 64
consultas. Cada trial = autoexp.trial con recon=dip sobre los g de desarrollo
(-4.0, -2.0, -0.5), los 3 en paralelo en la GPU del worker.

Objetivo (a maximizar): (mean_psnr + min_psnr) / 2 -- empuja el promedio sin dejar
tirado al g mas duro. Ambos quedan como user_attrs.

Varios workers comparten el estudio via JournalFileBackend (seguro en NFS):
    python -m autoexp.optuna_worker --study autoexp_v1 --hours 10

En el cluster lo lanza slurm/autoexp_optuna.slurm (array = workers).
"""
import argparse
import os
import time
import traceback

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from autoexp import trial as T

HERE = os.path.dirname(os.path.abspath(__file__))

# Receta actual de la tesis (presentacion 17/9): grilla + REG_NOISE_STD=0.08, sin
# pseudo-puntos; y la mejor combinacion local sin DIP (adaptativo 48+16) con pseudo-puntos.
SEEDS = [
    {"sampler": "grid", "aug": "none", "iters": 8000, "lr": 1e-3, "reg": 0.08,
     "input_type": "noise", "input_depth": 32, "width": 128, "scales": 5},
    {"sampler": "adaptive", "n1": 48, "power": 1.0, "aug": "plateau", "n_pseudo": 1024,
     "grad_q": 0.6, "iters": 8000, "lr": 1e-3, "reg": 0.08,
     "input_type": "noise", "input_depth": 32, "width": 128, "scales": 5},
]


def build_params(t):
    p = {"n": 64, "recon": "dip"}
    s = t.suggest_categorical("sampler", ["grid", "adaptive"])
    p["sampler"] = s
    if s == "grid":
        nx = t.suggest_categorical("grid_nx", [4, 6, 8, 10, 12, 16])
        p["sampler_kw"] = {"nx": nx, "offset": t.suggest_float("grid_offset", 0.2, 0.8)}
    else:
        p["sampler_kw"] = {"n1": t.suggest_int("n1", 24, 56, step=8),
                           "power": t.suggest_float("power", 0.5, 2.0)}

    aug = t.suggest_categorical("aug", ["none", "plateau", "all"])
    if aug != "none":
        p["aug"] = {"where": aug, "n_pseudo": t.suggest_int("n_pseudo", 64, 4096, log=True),
                    "grad_q": t.suggest_float("grad_q", 0.3, 0.9) if aug == "plateau" else 0.5}

    it = t.suggest_categorical("input_type", ["noise", "meshgrid"])
    d = {
        "NUM_ITER": t.suggest_int("iters", 2000, 12000, step=2000),
        "LR": t.suggest_float("lr", 1e-4, 1e-2, log=True),
        "REG_NOISE_STD": t.suggest_float("reg", 0.0, 0.2),
        "INPUT_TYPE": it,
        "NET_WIDTH": t.suggest_categorical("width", [32, 64, 128]),
        "NUM_SCALES": t.suggest_categorical("scales", [3, 4, 5]),
    }
    if it == "noise":
        d["INPUT_DEPTH"] = t.suggest_categorical("input_depth", [8, 32])
    p["dip"] = d
    return p


def objective(t, study_name):
    p = build_params(t)
    name = "%s_t%04d" % (study_name, t.number)
    s = T.run(name, p, T.DEV_GS, note="optuna %s trial %d" % (study_name, t.number), log=False,
              runs_dir=os.path.join(HERE, "runs", study_name))
    t.set_user_attr("mean_psnr", s["mean_psnr"])
    t.set_user_attr("min_psnr", s["min_psnr"])
    t.set_user_attr("mean_ssim", s["mean_ssim"])
    t.set_user_attr("per_g", {str(r["g"]): round(r["psnr"], 2) for r in s["per_g"]})
    return 0.5 * (s["mean_psnr"] + s["min_psnr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="autoexp_v1")
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--trial-minutes", type=float, default=25.0, help="margen para no cortar un trial")
    a = ap.parse_args()

    storage = JournalStorage(JournalFileBackend(os.path.join(HERE, "optuna_%s.log" % a.study)))
    study = optuna.create_study(study_name=a.study, storage=storage, direction="maximize",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(multivariate=True, group=True,
                                                                   n_startup_trials=12))
    if os.environ.get("SLURM_ARRAY_TASK_ID", "0") == "0" and len(study.trials) == 0:
        for s in SEEDS:
            study.enqueue_trial(s, skip_if_exists=True)

    deadline = time.time() + a.hours * 3600
    while time.time() + a.trial_minutes * 60 < deadline:
        try:
            study.optimize(lambda t: objective(t, a.study), n_trials=1, catch=(RuntimeError,))
        except Exception:
            traceback.print_exc()
        b = study.best_trial
        print("[%s] trials=%d  mejor #%d: %.2f (mean %.2f, min %.2f)" % (
            time.strftime("%H:%M"), len(study.trials), b.number, b.value,
            b.user_attrs.get("mean_psnr", -1), b.user_attrs.get("min_psnr", -1)), flush=True)


if __name__ == "__main__":
    main()
