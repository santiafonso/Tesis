#!/bin/bash
#SBATCH --job-name=dip_regnoise_fam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-26%12
#SBATCH --output=logs/dip_regnoise_fam_%A_%a.out
#SBATCH --error=logs/dip_regnoise_fam_%A_%a.err

# Comparacion para la presentacion del 16/9: como cambia la reconstruccion
# segun REG_NOISE_STD, para las 3 mejores familias de mascara (grid, uniform,
# frontier_mix), en 3 g representativos del barrido (-4.0 duro, -2.0 medio,
# -0.5 suave -- los mismos 3 ya usados en las demas diapositivas).
#
#   3 g x 3 familias x 3 REG_NOISE_STD = 27 tareas; %12 simultaneas.
#
#   lee  data/restoration/masks_g/g<g>/mf0.980/mask_<name>.npy
#        (ya generadas por slurm/frontier_mask_g.slurm)
#        results/phase_diagram_g/g<g>/sim_128.png
#   escribe results/experimentos_menos_puntos/barrido_reg_noise_familias/g<g>/mf0.980/<name>/reg<r>/
#
# Idempotente: salta si .../restored.npy existe. Borrar la carpeta para reforzar.
#
# Submit:  sbatch slurm/dip_reg_noise_familias.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
MF="${MF:-0.980}"
GS=(-4.0 -2.0 -0.5)
FAMILIES=(grid uniform frontier_mix)
REGS=(0.01 0.03 0.08)

NF=${#FAMILIES[@]}
NR=${#REGS[@]}
PER_G=$((NF * NR))

i=${SLURM_ARRAY_TASK_ID:-0}
g=${GS[$((i / PER_G))]}
rem=$((i % PER_G))
name=${FAMILIES[$((rem / NR))]}
reg=${REGS[$((rem % NR))]}

export PYTHONUNBUFFERED=1
IMAGE_PATH="results/phase_diagram_g/g${g}/sim_${RES}.png"
[ -f "$IMAGE_PATH" ] || IMAGE_PATH="data/restoration/phase_diagram_g/g${g}/sim_${RES}.png"
export IMAGE_PATH
export MASK_PATH="data/restoration/masks_g/g${g}/mf${MF}/mask_${name}.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="$reg"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/experimentos_menos_puntos/barrido_reg_noise_familias/g${g}/mf${MF}/${name}/reg${reg}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH  (correr antes slurm/phase_diagram_g.slurm)"
    exit 1
fi
if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH  (correr antes slurm/frontier_mask_g.slurm)"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  ->  g=$g  familia=$name  reg_noise=$reg  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
