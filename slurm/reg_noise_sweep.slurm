#!/bin/bash
#SBATCH --job-name=reg_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-26%12
#SBATCH --output=logs/reg_sweep_%A_%a.out
#SBATCH --error=logs/reg_sweep_%A_%a.err

# Barrido de REG_NOISE_STD en los 9 g negativos: la prueba de la ventana
# chica (window_square) mostro que REG_NOISE_STD=0.01 (afinada para el
# frente abrupto en la imagen completa) sobreajusta a manchas cuando hay
# pocos puntos en poco espacio. Este barrido chequea si el mismo problema
# aparece tambien en la imagen COMPLETA (128x128, no la ventana), para
# los distintos g y los 3 valores de REG_NOISE_STD ya probados.
#
#   9 g x 3 REG_NOISE_STD = 27 tareas, familia grid, mf=0.980 fijo
#   lee  data/restoration/masks_g/g<g>/mf0.980/mask_grid.npy
#   escribe results/reg_noise_sweep/g<g>/reg<val>/
#
# Submit:  sbatch slurm/reg_noise_sweep.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0)
REGS=(0.01 0.03 0.08)

NR=${#REGS[@]}
i=${SLURM_ARRAY_TASK_ID:-0}
g=${GS[$((i / NR))]}
reg=${REGS[$((i % NR))]}

export PYTHONUNBUFFERED=1
IMAGE_PATH="results/phase_diagram_g/g${g}/sim_${RES}.png"
[ -f "$IMAGE_PATH" ] || IMAGE_PATH="data/restoration/phase_diagram_g/g${g}/sim_${RES}.png"
export IMAGE_PATH
export MASK_PATH="data/restoration/masks_g/g${g}/mf0.980/mask_grid.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="$reg"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/reg_noise_sweep/g${g}/reg${reg}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH"
    exit 1
fi
if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i -> g=$g REG_NOISE_STD=$reg ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
