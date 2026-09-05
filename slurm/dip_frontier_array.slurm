#!/bin/bash
#SBATCH --job-name=dip_frontier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-23%12
#SBATCH --output=logs/dip_frontier_%A_%a.out
#SBATCH --error=logs/dip_frontier_%A_%a.err

# Variante job-array de slurm/dip_frontier.slurm para la particion `short`
# (limite 1 h, mucha menos cola que `multi`): una corrida (mf, mask) por tarea.
# 6 mf x 4 mascaras = 24 tareas; %12 = hasta 12 simultaneas.
#
# Mismas entradas/salidas que dip_frontier.slurm:
#   lee  data/restoration/masks/mf<frac>/mask_<name>.npy
#   escribe results/dip_frontier/mf<frac>/<name>/
#
# Si alguna tarea corta por tiempo (TIMEOUT), reenviar esas a `multi`:
#   MFS="0.999" MASKS="spread" sbatch --export=ALL slurm/dip_frontier.slurm
#
# Submit:  sbatch slurm/dip_frontier_array.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
MFS=(0.980 0.990 0.995 0.997 0.998 0.999)
MASKS=(uniform frontier frontier_mix spread)

i=${SLURM_ARRAY_TASK_ID:-0}
mf=${MFS[$((i / 4))]}
name=${MASKS[$((i % 4))]}

export PYTHONUNBUFFERED=1
export IMAGE_PATH="./data/restoration/phase_diagram_sim_${RES}.png"
export NUM_ITER="${NUM_ITER:-11000}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SEED="${SEED:-42}"
export MASK_PATH="data/restoration/masks/mf${mf}/mask_${name}.npy"
export OUTPUT_DIR="results/dip_frontier/mf${mf}/${name}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH"
    exit 1
fi

echo "=== array task $i  ->  mf=$mf  mask=$name  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
