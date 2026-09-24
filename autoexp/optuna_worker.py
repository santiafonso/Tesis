"""Worker de Optuna (puntos 1-3 de la reunion del 17/9) -- MODIFICABLE por el loop.

Busca pseudo-puntos del modelo de acantilado + hiperparametros de DIP, con el muestreo
del loop local (biseccion del acantilado) y presupuesto fijo de 64 consultas. Cada trial = autoexp.trial con recon=dip sobre los g de desarrollo
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

# v1 (23/9, reformulado antes de arrancar): el muestreo queda fijo en lo mejor del loop local
# (grilla 6x6 + biseccion del acantilado a 0 + relleno adaptativo; clasico solo = 34.4 dB en
# los 9 g). La pregunta es si DIP mejora la parte suave con pseudo-puntos del modelo de
# acantilado (ceros debajo del borde, opcionalmente mesetas de la reconstruccion clasica).
SEEDS = [
    {"n1": 30, "aug": "zero", "n_zero": 4096, "post_zero": True, "iters": 8000, "lr": 1e-3,
     "reg": 0.08, "input_type": "noise", "input_depth": 32, "width": 128, "scales": 5, "skip": 4,
     "upsample": "bilinear"},
    {"n1": 30, "aug": "zero+plateau", "n_zero": 4096, "n_pseudo": 256, "grad_q": 0.5,
     "post_zero": True, "iters": 8000, "lr": 1e-3, "reg": 0.08, "input_type": "noise",
     "input_depth": 32, "width": 128, "scales": 5, "skip": 4, "upsample": "bilinear"},
]


def build_params(t):
    p = {"n": 64, "recon": "dip", "sampler": "bisect",
         "sampler_kw": {"n1": t.suggest_categorical("n1", [24, 30, 36]), "nx": 6, "target": "zero",
                        "fill": "loo", "batch": 4}}
    aug = t.suggest_categorical("aug", ["none", "zero", "zero+plateau"])
    if aug != "none":
        p["aug"] = {"where": aug, "n_zero": t.suggest_int("n_zero", 256, 8192, log=True)}
        if aug == "zero+plateau":
            p["aug"]["n_pseudo"] = t.suggest_int("n_pseudo", 32, 1024, log=True)
            p["aug"]["grad_q"] = t.suggest_float("grad_q", 0.2, 0.8)
        p["post_zero"] = t.suggest_categorical("post_zero", [True, False])

    p["dip"] = dip_space(t)
    return p


def dip_space(t):
    """Hiperparametros de DIP. Incluye los del paper (notebooks/inpainting.ipynb):
    vase (agujeros grandes): meshgrid, LR 0.01, 5001 it, reg 0.03, skip 0, nearest;
    kate/peppers: noise 32, LR 0.01, 6001 it, reg 0.03, skip 128, nearest.
    La tesis venia usando LR 0.001, skip 4, bilinear, reg 0.08."""
    it = t.suggest_categorical("input_type", ["noise", "meshgrid"])
    d = {
        "NUM_ITER": t.suggest_int("iters", 2000, 12000, step=1000),
        "LR": t.suggest_float("lr", 1e-4, 2e-2, log=True),
        "REG_NOISE_STD": t.suggest_float("reg", 0.0, 0.2),
        "INPUT_TYPE": it,
        "NET_WIDTH": t.suggest_categorical("width", [32, 64, 128]),
        "NUM_SCALES": t.suggest_categorical("scales", [3, 4, 5]),
        "SKIP_N11": t.suggest_categorical("skip", [0, 4, 16, 128]),
        "UPSAMPLE_MODE": t.suggest_categorical("upsample", ["bilinear", "nearest"]),
    }
    if it == "noise":
        d["INPUT_DEPTH"] = t.suggest_categorical("input_depth", [8, 32])
    return d


# Pedido de los profes (17/9): los mismos experimentos para grid y uniform tocando
# hiperparametros, con la grilla como un hiperparametro mas. Solo DIP, sin pseudo-puntos.
_PAPER = {"lr": 0.01, "reg": 0.03, "width": 128, "scales": 5, "upsample": "nearest"}
PROFES_SEEDS = [
    dict(_PAPER, sampler="grid", grid_nx=8, grid_offset=0.5, input_type="meshgrid", iters=5000, skip=0),  # vase
    dict(_PAPER, sampler="grid", grid_nx=8, grid_offset=0.5, input_type="noise", input_depth=32,
         iters=6000, skip=128),  # kate/peppers
    dict(_PAPER, sampler="uniform", uniform_seed=0, input_type="meshgrid", iters=5000, skip=0),
    {"sampler": "grid", "grid_nx": 8, "grid_offset": 0.5, "input_type": "noise", "input_depth": 32,
     "iters": 8000, "lr": 1e-3, "reg": 0.08, "width": 128, "scales": 5, "skip": 4,
     "upsample": "bilinear"},  # receta de la tesis (17/9)
]


def build_params_profes(t):
    p = {"n": 64, "recon": "dip"}
    s = t.suggest_categorical("sampler", ["grid", "uniform"])
    p["sampler"] = s
    if s == "grid":
        p["sampler_kw"] = {"nx": t.suggest_categorical("grid_nx", [4, 6, 8, 10, 12, 16]),
                           "offset": t.suggest_float("grid_offset", 0.2, 0.8)}
    else:
        p["sampler_kw"] = {"seed": t.suggest_int("uniform_seed", 0, 9)}
    p["dip"] = dip_space(t)
    return p


def objective(t, study_name):
    p = build_params_profes(t) if study_name.startswith("profes") else build_params(t)
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
    # semillas: se encolan las que no tengan un trial vivo o completo (una semilla que fallo
    # por un bug ya corregido se vuelve a encolar)
    alive = [t.params for t in study.get_trials(deepcopy=False, states=(
        optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING))]
    for s in (PROFES_SEEDS if a.study.startswith("profes") else SEEDS):
        if not any(all(p.get(k) == v for k, v in s.items()) for p in alive):
            study.enqueue_trial(s)

    deadline = time.time() + a.hours * 3600
    while time.time() + a.trial_minutes * 60 < deadline:
        try:
            study.optimize(lambda t: objective(t, a.study), n_trials=1, catch=(RuntimeError,))
        except Exception:
            traceback.print_exc()
        try:
            b = study.best_trial
            print("[%s] trials=%d  mejor #%d: %.2f (mean %.2f, min %.2f)" % (
                time.strftime("%H:%M"), len(study.trials), b.number, b.value,
                b.user_attrs.get("mean_psnr", -1), b.user_attrs.get("min_psnr", -1)), flush=True)
        except ValueError:
            print("[%s] trials=%d, ninguno completo todavia" % (time.strftime("%H:%M"), len(study.trials)), flush=True)


if __name__ == "__main__":
    main()
