#!/bin/bash
#SBATCH --job-name=dip_fm_reg008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=logs/dip_fm_reg008_%A_%a.out
#SBATCH --error=logs/dip_fm_reg008_%A_%a.err

# Completar la tabla: frontier_mix + REG_NOISE_STD=0.08 (el mejor metodo
# encontrado el 15/9) en los 6 g que faltaban (ya se corrio -4.0,-2.0,-0.5
# en barrido_reg_noise_familias). 328 pts, mf=0.980.
#
# Submit:  sbatch slurm/dip_frontier_mix_reg008_allg.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

GS=(-3.5 -3.0 -2.5 -1.5 -1.0 0.0)
MF="0.980"
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
export OUTPUT_DIR="results/experimentos_menos_puntos/barrido_reg_noise_familias/g${g}/mf${MF}/${FAMILY}/reg0.08"

mkdir -p logs "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i (g=$g)  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
