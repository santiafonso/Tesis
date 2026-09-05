#!/bin/bash
#SBATCH --job-name=dip_frontier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dip_frontier_%j.out
#SBATCH --error=logs/dip_frontier_%j.err

# Prueba "puntos en la frontera vs sin puntos en la frontera", barrida ademas en
# el % de puntos observados. Para cada (mf, distribucion) misma cantidad total N
# de pixeles observados, distinta ubicacion espacial:
#   uniform      : baseline Bernoulli
#   frontier     : ~todos los puntos sobre la frontera de fase (alto |grad SoC|)
#   frontier_mix : mayoria sobre la frontera + una fraccion repartida afuera
#   spread       : los puntos evitan la frontera (mesetas)
#
# Las mascaras se generan antes, local, con:
#   ./venv/bin/python -m dip.frontier_mask
# y quedan versionadas en data/restoration/masks/mf<frac>/mask_<name>.npy
#
# Salida (espeja a dip_sweep.slurm):
#   results/dip_frontier/mf<frac>/<name>/
#     mask.png iter_*.png final_comparison.png comparison_annotated.png
#     psnr_curve.png metrics.csv restored.png/.npy original.npy
#
# Submit:  sbatch slurm/dip_frontier.slurm
#          MFS="0.990 0.999" MASKS="frontier spread" sbatch --export=ALL slurm/dip_frontier.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
MASK_DIR="${MASK_DIR:-data/restoration/masks}"
OUT_ROOT="${OUT_ROOT:-results/dip_frontier}"

mkdir -p logs "$OUT_ROOT"

export PYTHONUNBUFFERED=1
export IMAGE_PATH="./data/restoration/phase_diagram_sim_${RES}.png"
export NUM_ITER="${NUM_ITER:-11000}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SEED="${SEED:-42}"

MFS="${MFS:-0.980 0.990 0.995 0.997 0.998 0.999}"
MASKS="${MASKS:-uniform frontier frontier_mix spread}"

for mf in $MFS; do
    for name in $MASKS; do
        export MASK_PATH="${MASK_DIR}/mf${mf}/mask_${name}.npy"
        export OUTPUT_DIR="${OUT_ROOT}/mf${mf}/${name}"
        if [ ! -f "$MASK_PATH" ]; then
            echo "!! falta $MASK_PATH -- salteo"
            continue
        fi
        echo "==================================================================="
        echo "mf=${mf}  mask=${name}  ->  ${OUTPUT_DIR}   ($(date))"
        echo "==================================================================="
        ./venv/bin/python -m dip.restoration
    done
done

echo "Prueba frontera completa ($(date)). Resultados en ${OUT_ROOT}/"
