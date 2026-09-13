#!/bin/bash
#SBATCH --job-name=frontier_mask_g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-10%11
#SBATCH --output=logs/frontier_mask_g_%A_%a.out
#SBATCH --error=logs/frontier_mask_g_%A_%a.err

# Genera mascaras de frontera POR VALOR DE g (una tarea por g). Es el paso previo
# a slurm/dip_gsweep_frontier.slurm: para los mapas con frente de fase abrupto
# (g negativos) conviene concentrar los pixeles observados sobre la frontera en
# vez de repartirlos Bernoulli-uniforme.
#
#   lee  results/phase_diagram_g/g<g>/{sim_<RES>.png, soc.npy}
#        (fallback: data/restoration/phase_diagram_g/g<g>/...)
#   escribe data/restoration/masks_g/g<g>/mf<frac>/mask_{frontier,frontier_mix,spread,uniform}.npy
#
# OBS_FRACS espeja los MASK_FRAC de dip_gsweep_frontier.slurm (0.90..0.995).
# Idempotente: salta si ya existe la mascara mas rala.
#
# frontier_mask.py necesita Python moderno con scipy (galpynostatic no hace falta
# aca). En Mendieta:
#   PYBIN=~/micromamba/envs/galpy/bin/python sbatch --export=ALL slurm/frontier_mask_g.slurm
# En local PYBIN queda en ./venv/bin/python.
#
# Submit:  PYBIN=~/micromamba/envs/galpy/bin/python sbatch --export=ALL slurm/frontier_mask_g.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

PYBIN="${PYBIN:-./venv/bin/python}"
RES="${RES:-128}"
GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 2.0 4.0)
g=${GS[${SLURM_ARRAY_TASK_ID:-0}]}

OBS_FRACS="${OBS_FRACS:-0.10 0.05 0.02 0.01 0.005}"   # -> MASK_FRAC 0.90 .. 0.995
OUT_DIR="data/restoration/masks_g/g${g}"

IMAGE_PATH="results/phase_diagram_g/g${g}/sim_${RES}.png"
[ -f "$IMAGE_PATH" ] || IMAGE_PATH="data/restoration/phase_diagram_g/g${g}/sim_${RES}.png"
SOC_NPY="results/phase_diagram_g/g${g}/soc.npy"
[ -f "$SOC_NPY" ] || SOC_NPY="data/restoration/phase_diagram_g/g${g}/soc.npy"

mkdir -p logs "$OUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH  (correr antes slurm/phase_diagram_g.slurm)"
    exit 1
fi

# mf de la fraccion observada mas chica -> mascara "mas rala" (marca idempotencia)
MIN_OBS=$(echo "$OBS_FRACS" | tr ' ' '\n' | sort -g | head -1)
MF_RALA=$(awk "BEGIN{printf \"%.3f\", 1 - $MIN_OBS}")
if [ -f "$OUT_DIR/mf${MF_RALA}/mask_frontier_mix.npy" ]; then
    echo "== ya hecho: $OUT_DIR (borrar para reforzar) =="
    exit 0
fi

export PYTHONUNBUFFERED=1
export IMAGE_PATH SOC_NPY OUT_DIR OBS_FRACS
export SEED="${SEED:-42}"
# 0.40 (vs. el 0.15 por defecto): a mf alto la familia frontier casi no deja
# puntos en las mesetas y sus niveles derivan -> se le sube el termino uniforme.
export MIX_FRAC="${MIX_FRAC:-0.40}"

echo "=== g=$g  RES=$RES  OBS_FRACS='$OBS_FRACS'  ($(date)) ==="
"$PYBIN" -m dip.frontier_mask
echo "=== done g=$g ($(date)) ==="
