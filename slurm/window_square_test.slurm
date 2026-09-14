#!/bin/bash
#SBATCH --job-name=win_sq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=logs/win_sq_%j.out
#SBATCH --error=logs/win_sq_%j.err

# Ventana cuadrada 64x64 (el minimo que acepta dip.restoration) centrada en
# el "codo" de la frontera (maxima curvatura), sin enderezar -- recorte
# axis-aligned directo. Preparado por analysis/window_square.py prepare;
# se empalma despues en local con analysis/window_square.py splice.
#
#   lee  results/dip_gsweep_frontier/g-4.0/mf0.980/grid/window_sq64/{image_window.png,mask_window.npy}
#   escribe results/dip_gsweep_frontier/g-4.0/mf0.980/grid/window_sq64_dip/
#
# Submit:  sbatch slurm/window_square_test.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RUN_DIR="results/dip_gsweep_frontier/g-4.0/mf0.980/grid"
export PYTHONUNBUFFERED=1
export IMAGE_PATH="${RUN_DIR}/window_sq64/image_window.png"
export MASK_PATH="${RUN_DIR}/window_sq64/mask_window.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="${REG_NOISE_STD:-0.08}"  # 0.01 (afinada p/ imagen completa) sobreajusta a manchas en ventanas chicas
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="${RUN_DIR}/window_sq64_dip_reg${REG_NOISE_STD}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ] || [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $IMAGE_PATH o $MASK_PATH (correr antes analysis/window_square.py prepare)"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== ventana cuadrada $RUN_DIR ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done ($(date)) ==="
