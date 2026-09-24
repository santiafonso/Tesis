#!/bin/bash
#SBATCH --job-name=dip_famaf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-6
#SBATCH --output=logs/dip_famaf_%A_%a.out
#SBATCH --error=logs/dip_famaf_%A_%a.err

# A modo de ejemplo (15/9): foto real (edificio de FAMAF, RGB, 1024x768) a
# los mismos % observado que la tanda sintetica, para comparar contra los
# graficos matematicos y contra el diagrama de fase.
#
# Submit:  sbatch slurm/dip_edificio_famaf.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

OBS_PCTS=(1 5 10 20 30 40 50)

i=${SLURM_ARRAY_TASK_ID:-0}
pct=${OBS_PCTS[$i]}
mf=$(echo "scale=4; 1 - $pct/100" | bc)

export PYTHONUNBUFFERED=1
export IMAGE_PATH="data/restoration/edificio_famaf.jpg"
export MASK_FRAC="$mf"
export N_CHANNELS=3
export MAX_SIDE="${MAX_SIDE:-512}"
export NUM_ITER="${NUM_ITER:-8000}"
export REG_NOISE_STD="${REG_NOISE_STD:-0.03}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/synthetic_targets_demo/edificio_famaf/obs${pct}pct"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  obs=${pct}%  MASK_FRAC=$mf  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
