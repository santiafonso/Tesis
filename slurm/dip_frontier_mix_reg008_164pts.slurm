#!/bin/bash
#SBATCH --job-name=dip_fm_164
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/dip_fm_164_%A_%a.out
#SBATCH --error=logs/dip_fm_164_%A_%a.err

# Punto intermedio que faltaba: frontier_mix + REG_NOISE_STD=0.08 a 164 pts
# (mf=0.990, 1% obs.), en los 9 g -- entre los 82 pts (fewpoints_3g, solo 3
# g) y los 328 pts (barrido_reg_noise_familias, los 9 g).
#
# Submit:  sbatch slurm/dip_frontier_mix_reg008_164pts.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0)
MF="0.990"
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
