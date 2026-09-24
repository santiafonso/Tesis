#!/bin/bash
#SBATCH --job-name=dip_lr_reg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=logs/dip_lr_reg_%A_%a.out
#SBATCH --error=logs/dip_lr_reg_%A_%a.err

# Objetivo (16/9): ver si tocar LR (aparte de REG_NOISE_STD/NUM_ITER, ya
# explorados) permite subir el PSNR en el regimen de POCOS puntos, en vez de
# necesitar mas puntos. Caso fijo: g=-4.0 (el mas duro), familia grid,
# mf=0.995 (~82 pts, el mas disperso ya corrido -- baseline LR=0.001/
# REG_NOISE_STD=0.01: PSNR 19.3 dB, SSIM 0.609, en
# results/dip_gsweep_frontier/g-4.0/mf0.995/grid/).
#
# 5 combinaciones (no cartesiano completo, elegidas a mano):
#   0: LR=0.0005 REG=0.01   (mas lento/cuidadoso, mismo ruido)
#   1: LR=0.002  REG=0.01   (mas rapido, mismo ruido)
#   2: LR=0.001  REG=0.005  (menos ruido de regularizacion)
#   3: LR=0.0005 REG=0.005  (combo clasico DIP: LR bajo + reg bajo -> bordes nitidos)
#   4: LR=0.001  REG=0.03   (mas ruido, para contrastar en el otro extremo)
#   5: LR=0.001  REG=0.08   (el valor que mejor anduvo en el otro barrido, con
#      328 pts -- no se habia probado todavia aca con solo 82)
#
# Submit:  sbatch slurm/dip_lr_regnoise_sweep.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

G="-4.0"
MF="0.995"
FAMILY="grid"
LRS=(0.0005 0.002  0.001  0.0005 0.001 0.001)
REGS=(0.01   0.01   0.005  0.005  0.03  0.08)

i=${SLURM_ARRAY_TASK_ID:-0}
lr=${LRS[$i]}
reg=${REGS[$i]}

export PYTHONUNBUFFERED=1
export IMAGE_PATH="results/phase_diagram_g/g${G}/sim_128.png"
export MASK_PATH="data/restoration/masks_g/g${G}/mf${MF}/mask_${FAMILY}.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export LR="$lr"
export REG_NOISE_STD="$reg"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/experimentos_menos_puntos/lr_regnoise_sweep/g${G}/mf${MF}/${FAMILY}/lr${lr}_reg${reg}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  ->  LR=$lr  REG_NOISE_STD=$reg  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
