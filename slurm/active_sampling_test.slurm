#!/bin/bash
#SBATCH --job-name=active_smpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/active_smpl_%A_%a.out
#SBATCH --error=logs/active_smpl_%A_%a.err

# Prototipo: compara la mascara "curve_trace" (puntos parejos por longitud de
# arco sobre la frontera real + mesetas, analysis/curve_trace_mask.py) contra
# uniform/grid/frontier_mix, las 4 al MISMO N, en g=-4.0 (el caso mas duro).
# Dos presupuestos de N (0-3 = ~48, 4-7 = ~120) para ver si el resultado
# depende de cuantos puntos hay. Mismos hiperparametros que
# dip_gsweep_frontier.slurm (g negativo).
#
#   lee  data/restoration/masks_g/g-4.0/mf0.997/mask_{uniform,grid,frontier_mix}.npy   (N~48)
#        data/restoration/masks_g/g-4.0/mf0.993/mask_{uniform,grid,frontier_mix}.npy   (N~120)
#        data/restoration/masks_active/g-4.0/mask_curve_trace_N{48,120}.npy
#   escribe results/active_sampling/g-4.0/<family>_N<n>/
#
# Submit:  sbatch slurm/active_sampling_test.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

NAMES=(uniform grid frontier_mix curve_trace uniform grid frontier_mix curve_trace)
NS=(48 48 48 48 120 120 120 120)
MFS=(0.997 0.997 0.997 0.997 0.993 0.993 0.993 0.993)
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
export REG_NOISE_STD="${REG_NOISE_STD:-0.01}"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
export SEED="${SEED:-42}"
if [ "$n" = "48" ]; then
  export OUTPUT_DIR="results/active_sampling/g-4.0/${name}"  # ya corridas antes, sin sufijo
else
  export OUTPUT_DIR="results/active_sampling/g-4.0/${name}_N${n}"
fi

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i -> $name  MASK_PATH=$MASK_PATH  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
