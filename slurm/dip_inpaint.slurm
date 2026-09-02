#!/bin/bash
#SBATCH --job-name=dip_inpaint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dip_inpaint_%j.out
#SBATCH --error=logs/dip_inpaint_%j.err

# Una corrida de inpainting DIP sobre una sola imagen/mascara (dip.inpaint).
# Reemplaza al viejo job.sh.
#
# La ruta al repo se toma de REPO_DIR, si no del dir desde el que se hizo sbatch,
# si no del cwd -- sirve igual para tu maquina y para la cuenta del cluster sin
# editar nada.
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

mkdir -p logs

export PYTHONUNBUFFERED=1

# Parametros (ver dip/inpaint.py para la lista completa y los defaults):
export IMAGE_PATH="${IMAGE_PATH:-./data/restoration/phase_diagram_sim_128.png}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/dip_inpaint}"
export MASK_FRAC="${MASK_FRAC:-0.90}"
export NUM_ITER="${NUM_ITER:-11000}"
export SHOW_EVERY="${SHOW_EVERY:-100}"

./venv/bin/python -m dip.inpaint
