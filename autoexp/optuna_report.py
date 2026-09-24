"""Resumen de un estudio Optuna de autoexp: tabla de trials + figura.

Uso (donde este el log del estudio, ej. en Mendieta o tras traerlo con rsync):
    python -m autoexp.optuna_report autoexp_v1 [profes_grid_uniform ...]

Escribe autoexp/runs/<estudio>/report.csv y report.png (media vs peor g por trial,
coloreado por sampler/aug, y la evolucion del mejor).
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

HERE = os.path.dirname(os.path.abspath(__file__))


def report(name):
    st = optuna.load_study(study_name=name, storage=JournalStorage(
        JournalFileBackend(os.path.join(HERE, "optuna_%s.log" % name))))
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    out_dir = os.path.join(HERE, "runs", name)
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted({k for t in done for k in t.params})
    with open(os.path.join(out_dir, "report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "valor", "mean_psnr", "min_psnr", "per_g"] + keys)
        for t in sorted(done, key=lambda t: -t.value):
            w.writerow([t.number, "%.2f" % t.value, "%.2f" % t.user_attrs["mean_psnr"],
                        "%.2f" % t.user_attrs["min_psnr"], t.user_attrs.get("per_g")]
                       + [t.params.get(k, "") for k in keys])
    print("== %s: %d completos, %d fallidos, %d corriendo" % (
        name, len(done), sum(t.state == optuna.trial.TrialState.FAIL for t in st.trials),
        sum(t.state == optuna.trial.TrialState.RUNNING for t in st.trials)))
    for t in sorted(done, key=lambda t: -t.value)[:8]:
        print("  #%-4d mean %6.2f  min %6.2f  %s  | %s" % (
            t.number, t.user_attrs["mean_psnr"], t.user_attrs["min_psnr"], t.user_attrs.get("per_g"),
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.params.items()}))
    if not done:
        return

    grp = "aug" if "aug" in keys else "sampler"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for lab in sorted({str(t.params.get(grp)) for t in done}):
        ts = [t for t in done if str(t.params.get(grp)) == lab]
        ax[0].scatter([t.user_attrs["mean_psnr"] for t in ts], [t.user_attrs["min_psnr"] for t in ts],
                      label="%s=%s" % (grp, lab), s=25)
    ax[0].axhline(38, color="k", ls=":", lw=1)
    ax[0].axvline(38, color="k", ls=":", lw=1)
    ax[0].set_xlabel("PSNR medio (g = -4, -2, -0.5) [dB]")
    ax[0].set_ylabel("PSNR del peor g [dB]")
    ax[0].legend(fontsize=8)
    ax[0].set_title("%s: cada punto es un trial (64 puntos)" % name)
    order = sorted(done, key=lambda t: t.number)
    best, cur = [], -1e9
    for t in order:
        cur = max(cur, t.value)
        best.append(cur)
    ax[1].plot([t.number for t in order], [t.value for t in order], "o", ms=3, alpha=.5, label="trial")
    ax[1].plot([t.number for t in order], best, "-", label="mejor hasta ahi")
    ax[1].set_xlabel("trial")
    ax[1].set_ylabel("(media + peor) / 2 [dB]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "report.png"), dpi=110)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["autoexp_v1", "profes_grid_uniform"]:
        report(n)
