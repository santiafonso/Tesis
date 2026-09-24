#!/bin/bash
#SBATCH --job-name=dip_g4_push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/dip_g4_push_%A_%a.out
#SBATCH --error=logs/dip_g4_push_%A_%a.err

# 17/9: hasta donde se puede empujar el caso mas duro (g=-4.0, frontier_mix,
# 328 pts) mas alla del reg=0.08/LR=0.001/iter=16000 ya validado (41.9 dB).
# Baseline para comparar: results/experimentos_menos_puntos/
# barrido_reg_noise_familias/g-4.0/mf0.980/frontier_mix/reg0.08/ (41.9 dB).
#
#   0: reg=0.10  LR=0.001  iter=16000  (mas ruido)
#   1: reg=0.12  LR=0.001  iter=16000  (mas ruido todavia)
#   2: reg=0.08  LR=0.002  iter=16000  (LR mas alto)
#   3: reg=0.08  LR=0.001  iter=24000  (mas iteraciones)
#   4: reg=0.10  LR=0.002  iter=24000  (los 3 juntos)
#
# Submit:  sbatch slurm/dip_g4_push_further.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

REGS=(0.10  0.12  0.08  0.08  0.10)
LRS=(0.001 0.001 0.002 0.001 0.002)
ITERS=(16000 16000 16000 24000 24000)

i=${SLURM_ARRAY_TASK_ID:-0}
reg=${REGS[$i]}
lr=${LRS[$i]}
iters=${ITERS[$i]}

export PYTHONUNBUFFERED=1
export IMAGE_PATH="results/phase_diagram_g/g-4.0/sim_128.png"
export MASK_PATH="data/restoration/masks_g/g-4.0/mf0.990/mask_frontier_mix.npy"
export NUM_ITER="$iters"
export LR="$lr"
export REG_NOISE_STD="$reg"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/experimentos_menos_puntos/g4_push_further_164pts/reg${reg}_lr${lr}_iter${iters}"

mkdir -p logs "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  reg=$reg lr=$lr iter=$iters  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
