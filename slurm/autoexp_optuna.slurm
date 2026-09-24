#!/bin/bash
#SBATCH --job-name=autoexp_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=10:30:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=logs/autoexp_opt_%A_%a.out
#SBATCH --error=logs/autoexp_opt_%A_%a.err

# 23/9: busqueda Optuna de muestreo + pseudo-puntos + hiperparametros de DIP con
# presupuesto de 64 consultas (ver autoexp/CLAUDE.md). Cada tarea del array es un
# worker del MISMO estudio (JournalFileBackend en autoexp/optuna_<STUDY>.log).
#
# Dos pythons: el worker (optuna, scipy moderno) corre en ~/venv-autoexp; DIP sigue
# en ./venv (py3.6 + torch cu113) via DIP_PY.
#
# Submit (desde el clon autoexp):  sbatch slurm/autoexp_optuna.slurm
#   STUDY=autoexp_v2 HOURS=4 sbatch --export=ALL slurm/autoexp_optuna.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1
mkdir -p logs

export PYTHONUNBUFFERED=1
export DIP_PY="${DIP_PY:-$PWD/venv/bin/python}"
WORKER_PY="${WORKER_PY:-$HOME/venv-autoexp/bin/python}"

echo "=== worker ${SLURM_ARRAY_TASK_ID:-0} study=${STUDY:-autoexp_v1} ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$WORKER_PY" -m autoexp.optuna_worker --study "${STUDY:-autoexp_v1}" --hours "${HOURS:-10}"
echo "=== fin ($(date)) ==="
