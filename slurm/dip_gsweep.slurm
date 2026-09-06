#!/bin/bash
#SBATCH --job-name=dip_gsweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-54%12
#SBATCH --output=logs/dip_gsweep_%A_%a.out
#SBATCH --error=logs/dip_gsweep_%A_%a.err

# Reconstruccion DIP sobre los mapas de continuo del barrido de g (isoterma de
# Frumkin). Para cada mapa (un g) se reconstruye con mascara Bernoulli uniforme a
# varias fracciones enmascaradas.
#   11 g x 5 MASK_FRAC = 55 tareas; %12 = hasta 12 simultaneas.
#
#   lee  results/phase_diagram_g/g<val>/sim_<RES>.png   (lo deja slurm/phase_diagram_g.slurm)
#        o, si no esta ahi, data/restoration/phase_diagram_g/g<val>/sim_<RES>.png
#   escribe results/dip_gsweep/g<val>/mf<frac>/
#
# Idempotente: salta si results/dip_gsweep/g<val>/mf<frac>/restored.npy existe.
#
# Submit:  sbatch slurm/dip_gsweep.slurm
#          # encadenado a la generacion de mapas:
#          sbatch --dependency=afterok:<jobid_phase_diagram_g> slurm/dip_gsweep.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 2.0 4.0)
MFS=(0.90 0.95 0.98 0.99 0.995)
NMF=${#MFS[@]}

i=${SLURM_ARRAY_TASK_ID:-0}
g=${GS[$((i / NMF))]}
mf=${MFS[$((i % NMF))]}

export PYTHONUNBUFFERED=1
IMAGE_PATH="results/phase_diagram_g/g${g}/sim_${RES}.png"
[ -f "$IMAGE_PATH" ] || IMAGE_PATH="data/restoration/phase_diagram_g/g${g}/sim_${RES}.png"
export IMAGE_PATH
export MASK_FRAC="$mf"
export NUM_ITER="${NUM_ITER:-11000}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/dip_gsweep/g${g}/mf${mf}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  ->  g=$g  MASK_FRAC=$mf  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
