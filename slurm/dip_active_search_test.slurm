#!/bin/bash
#SBATCH --job-name=dip_active_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=logs/dip_active_search_%A_%a.out
#SBATCH --error=logs/dip_active_search_%A_%a.err

# Prueba 15/9: busqueda binaria activa (analysis/active_binary_search_mask.py)
# -- ubica la frontera con ~120 consultas adaptativas (desplegable con KMC
# real, no mira el campo denso de antemano) + 150 puntos reales extra
# concentrados en la curva encontrada + 400 sinteticos de meseta inventados.
# g=-4.0. Compara, ambos con REG_NOISE_STD=0.08:
#   0: real_only  -- solo los 270 puntos reales (busqueda + extra), imagen real
#   1: augmented  -- 270 reales + 400 sinteticos, imagen compuesta
#
# Puntaje final: comparar restored.npy contra soc.npy real, no la compuesta
# -- ver analysis/frontier_synth_augment_score.py (reusar con los nombres
# real_only/augmented en vez de baseline/augmented).
#
# Submit:  sbatch slurm/dip_active_search_test.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

TAG="${TAG:-active_search}"
GS=(-2.0 -0.5)

i=${SLURM_ARRAY_TASK_ID:-0}
G=${GS[$((i / 2))]}
SEARCH_DIR="data/restoration/masks_g/g${G}/${TAG}"
i=$((i % 2))

export PYTHONUNBUFFERED=1
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="0.08"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-200}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2000}"
export SEED="${SEED:-42}"

if [ "$i" -eq 0 ]; then
    NAME="real_only"
    export IMAGE_PATH="results/phase_diagram_g/g${G}/sim_128.png"
    export MASK_PATH="${SEARCH_DIR}/mask_real_only.npy"
else
    NAME="augmented"
    export IMAGE_PATH="${SEARCH_DIR}/image_composite.png"
    export MASK_PATH="${SEARCH_DIR}/mask_augmented.npy"
fi
export OUTPUT_DIR="results/experimentos_menos_puntos/${TAG}/g${G}/${NAME}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ] || [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $IMAGE_PATH o $MASK_PATH"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i ($NAME)  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
