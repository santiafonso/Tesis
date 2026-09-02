#!/bin/bash
#SBATCH --job-name=dip_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dip_sweep_%j.out
#SBATCH --error=logs/dip_sweep_%j.err

# Barrido DIP sobre la imagen del modelo del continuo con distintas fracciones de
# mascara. Reemplaza a run_gris_sweep.slurm (64x64) y run_gris_sweep_128.slurm
# (128x128): elegis la resolucion con RES.
#
#   RES=64  -> data/restoration/phase_diagram_sim_64.png   -> results/dip_sweep_64/
#   RES=128 -> data/restoration/phase_diagram_sim_128.png  -> results/dip_sweep_128/  (default)
#
# Una carpeta por %:  results/dip_sweep_${RES}/mf0.XX/
#   mask.png  iter_00000..iter_NNNNN.png  final_comparison.png
#   comparison_annotated.png  psnr_curve.png  metrics.csv
#   restored.png/.npy  original.npy
#
# La imagen de entrada se genera aparte con  ./venv/bin/python -m dip.phase_diagram
# (NUM_XI=NUM_ELL=RES) y se sube al cluster; este job solo la consume.
#
# Submit:  sbatch slurm/dip_sweep.slurm            (128x128)
#          RES=64 sbatch --export=ALL slurm/dip_sweep.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
IMAGE_PATH="./data/restoration/phase_diagram_sim_${RES}.png"
SWEEP_DIR="results/dip_sweep_${RES}"

mkdir -p logs "$SWEEP_DIR"

export PYTHONUNBUFFERED=1
export IMAGE_PATH
export NUM_ITER="${NUM_ITER:-11000}"
export SHOW_EVERY="${SHOW_EVERY:-100}"

MASKS="${MASKS:-0.50 0.70 0.80 0.90 0.95 0.98 0.99}"

for mf in $MASKS; do
    export MASK_FRAC=$mf
    export OUTPUT_DIR="${SWEEP_DIR}/mf${mf}"
    echo "==================================================================="
    echo "RES=$RES  MASK_FRAC=$mf  ->  $OUTPUT_DIR   ($(date))"
    echo "==================================================================="
    ./venv/bin/python -m dip.inpaint
done

echo "Barrido ${RES}x${RES} completo ($(date)). Resultados en ${SWEEP_DIR}/"
