#!/bin/bash
#SBATCH --job-name=win32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=logs/win32_%j.out
#SBATCH --error=logs/win32_%j.err

# Ventana cuadrada 32x32 REAL (no redimensionada) centrada en el codo de la
# frontera, con 36 puntos concentrados adentro. Necesita
# dip/restoration.py con DIM_DIV_BY=32 (la red baja 5 escalas con stride 2,
# 2**5=32 es el minimo real -- 64 era mas conservador de lo necesario).
#
#   lee  results/dip_gsweep_frontier/g-4.0/mf0.980/grid/window_sq32_concentrated_N40/{image_window.png,mask_window.npy}
#   escribe results/dip_gsweep_frontier/g-4.0/mf0.980/grid/window_sq32_concentrated_N40_dip/
#
# Submit:  sbatch slurm/window_32_test.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RUN_DIR="results/dip_gsweep_frontier/g-4.0/mf0.980/grid"
PREP_DIR="${RUN_DIR}/window_sq32_concentrated_N40"
export PYTHONUNBUFFERED=1
export IMAGE_PATH="${PREP_DIR}/image_window.png"
export MASK_PATH="${PREP_DIR}/mask_window.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="${REG_NOISE_STD:-0.08}"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="${PREP_DIR}_dip"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ] || [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $IMAGE_PATH o $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== ventana 32x32 $RUN_DIR ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done ($(date)) ==="
