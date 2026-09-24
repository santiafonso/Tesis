#!/bin/bash
#SBATCH --job-name=autoexp_tr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=logs/autoexp_tr_%A_%a.out
#SBATCH --error=logs/autoexp_tr_%A_%a.err

# Corridas puntuales de autoexp.trial (no Optuna): una linea de $QUEUE por tarea del array,
# "nombre<TAB>params json[<TAB>gs]". gs por defecto: los 3 de desarrollo.
#   QUEUE=autoexp/queue/hibrido_1.txt sbatch --array=0-5 --export=ALL slurm/autoexp_trials.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1
export DIP_PY="${DIP_PY:-$PWD/venv/bin/python}"
line=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$QUEUE")
name=$(echo "$line" | cut -f1); params=$(echo "$line" | cut -f2); gs=$(echo "$line" | cut -f3)
echo "=== $name ($(date)) ==="
"${WORKER_PY:-$HOME/venv-autoexp/bin/python}" -m autoexp.trial "$name" "$params" --gs="${gs:--4.0,-2.0,-0.5}" --note "cluster $QUEUE"
echo "=== fin ($(date)) ==="
