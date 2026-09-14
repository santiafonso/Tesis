#!/bin/bash
#SBATCH --job-name=active_reg08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/active_reg08_%A_%a.out
#SBATCH --error=logs/active_reg08_%A_%a.err

# Repite el test de active_sampling_test.slurm (curve_trace/uniform/grid/
# frontier_mix, N~48 y N~120, g=-4.0) pero con REG_NOISE_STD=0.08 en vez de
# 0.01 -- la ventana chica mostro que 0.01 sobreajusta a manchas con pocos
# puntos; esto chequea si el mismo problema (y el mismo arreglo) aplica
# tambien en la imagen COMPLETA 128x128, no solo en un recorte.
#
#   lee las mismas mascaras que active_sampling_test.slurm
#   escribe results/active_sampling/g-4.0/<family>_N<n>_reg008/
#
# Submit:  sbatch slurm/active_sampling_reg008.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

NAMES=(uniform grid frontier_mix curve_trace uniform grid frontier_mix curve_trace)
NS=(48 48 48 48 120 120 120 120)
MASKS=(
  "data/restoration/masks_g/g-4.0/mf0.997/mask_uniform.npy"
  "data/restoration/masks_g/g-4.0/mf0.997/mask_grid.npy"
  "data/restoration/masks_g/g-4.0/mf0.997/mask_frontier_mix.npy"
  "data/restoration/masks_active/g-4.0/mask_curve_trace_N48.npy"
  "data/restoration/masks_g/g-4.0/mf0.993/mask_uniform.npy"
  "data/restoration/masks_g/g-4.0/mf0.993/mask_grid.npy"
  "data/restoration/masks_g/g-4.0/mf0.993/mask_frontier_mix.npy"
  "data/restoration/masks_active/g-4.0/mask_curve_trace_N120.npy"
)

i=${SLURM_ARRAY_TASK_ID:-0}
name=${NAMES[$i]}
n=${NS[$i]}
export PYTHONUNBUFFERED=1
export IMAGE_PATH="results/phase_diagram_g/g-4.0/sim_128.png"
export MASK_PATH="${MASKS[$i]}"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="${REG_NOISE_STD:-0.08}"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/active_sampling/g-4.0/${name}_N${n}_reg008"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i -> $name N=$n REG_NOISE_STD=$REG_NOISE_STD ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
