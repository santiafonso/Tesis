#!/bin/bash
#SBATCH --job-name=dip_synth_aug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=logs/dip_synth_aug_%A_%a.out
#SBATCH --error=logs/dip_synth_aug_%A_%a.err

# Prueba 15/9 (fuera de los 5 puntos): inventar puntos de meseta en vez de
# gastar presupuesto real ahi, concentrando el presupuesto REAL en la
# frontera. Caso: g=-4.0, frontier_mix, mf=0.995 (82 pts reales -- el mas
# disperso, donde frontier_mix sola colapsa a 8.1 dB por dejar la meseta sin
# puntos). Compara, ambos con REG_NOISE_STD=0.08 (el hallazgo de hoy):
#   0: baseline   -- solo los 82 puntos reales, imagen real
#   1: augmented  -- los mismos 82 reales + 400 sinteticos (valor inventado
#      por relajacion exponencial ajustada a esos mismos 82 puntos, ver
#      analysis/frontier_synth_augment.py), imagen compuesta
#
# El score final (PSNR/SSIM) hay que recalcularlo aparte contra el soc.npy
# real (no contra la imagen compuesta, que tiene valores inventados) -- ver
# analysis/frontier_synth_augment_score.py despues de que termine el job.
#
# Submit:  sbatch slurm/dip_synth_augment_test.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

G="-4.0"
MF="0.995"
FAMILY="frontier_mix"
SYNTH_DIR="data/restoration/masks_g/g${G}/mf${MF}/synth_${FAMILY}"

i=${SLURM_ARRAY_TASK_ID:-0}

export PYTHONUNBUFFERED=1
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="0.08"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"

if [ "$i" -eq 0 ]; then
    NAME="baseline"
    export IMAGE_PATH="results/phase_diagram_g/g${G}/sim_128.png"
    export MASK_PATH="data/restoration/masks_g/g${G}/mf${MF}/mask_${FAMILY}.npy"
else
    NAME="augmented"
    export IMAGE_PATH="${SYNTH_DIR}/image_composite.png"
    export MASK_PATH="${SYNTH_DIR}/mask_augmented.npy"
fi
export OUTPUT_DIR="results/experimentos_menos_puntos/synth_augment/g${G}/mf${MF}/${FAMILY}/${NAME}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ] || [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $IMAGE_PATH o $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i ($NAME)  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
