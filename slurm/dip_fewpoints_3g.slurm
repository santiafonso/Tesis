#!/bin/bash
#SBATCH --job-name=dip_fewpts_3g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/dip_fewpts_3g_%A_%a.out
#SBATCH --error=logs/dip_fewpts_3g_%A_%a.err

# Misma version que best_per_g.png (frontier_mix + REG_NOISE_STD=0.08) pero
# con MENOS puntos (mf=0.995, ~82 pts en vez de 328) para los 3 g de siempre,
# para ver como se degrada visualmente.
#
# Submit:  sbatch slurm/dip_fewpoints_3g.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

GS=(-4.0 -2.0 -0.5)
MF="0.995"
FAMILY="frontier_mix"

i=${SLURM_ARRAY_TASK_ID:-0}
g=${GS[$i]}

export PYTHONUNBUFFERED=1
export IMAGE_PATH="results/phase_diagram_g/g${g}/sim_128.png"
export MASK_PATH="data/restoration/masks_g/g${g}/mf${MF}/mask_${FAMILY}.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="0.08"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/experimentos_menos_puntos/fewpoints_3g/g${g}/mf${MF}/${FAMILY}"

mkdir -p logs "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i (g=$g)  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
