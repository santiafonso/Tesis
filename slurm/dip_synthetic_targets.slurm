#!/bin/bash
#SBATCH --job-name=dip_synth_targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-34%12
#SBATCH --output=logs/dip_synth_targets_%A_%a.out
#SBATCH --error=logs/dip_synth_targets_%A_%a.err

# A modo de ejemplo (15/9): correr dip.restoration sobre las 5 imagenes
# "de libro" (dip/synthetic_targets.py) a distintos % observado, para ver
# como reconstruye contenido que no es un diagrama de fase -- conecta la
# pregunta de "cuantos puntos hacen falta" con la frecuencia espacial del
# contenido en general, no solo con el caso particular del frente de fase.
#
#   5 imagenes x 7 % observado = 35 tareas; %12 simultaneas.
#
# Submit:  sbatch slurm/dip_synthetic_targets.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

IMAGES=(gauss freq_sweep rosenbrock himmelblau checkerboard)
OBS_PCTS=(1 5 10 20 30 40 50)

NP=${#OBS_PCTS[@]}

i=${SLURM_ARRAY_TASK_ID:-0}
img=${IMAGES[$((i / NP))]}
pct=${OBS_PCTS[$((i % NP))]}
mf=$(echo "scale=4; 1 - $pct/100" | bc)

export PYTHONUNBUFFERED=1
export IMAGE_PATH="data/restoration/synthetic/${img}.png"
export MASK_FRAC="$mf"
export N_CHANNELS=1
export NUM_ITER="${NUM_ITER:-8000}"
export REG_NOISE_STD="${REG_NOISE_STD:-0.03}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/synthetic_targets_demo/${img}/obs${pct}pct"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH  (correr antes ./venv/bin/python -m dip.synthetic_targets)"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  img=$img  obs=${pct}%  MASK_FRAC=$mf  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
