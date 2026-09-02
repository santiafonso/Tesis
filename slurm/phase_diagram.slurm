#!/bin/bash
#SBATCH --job-name=phase_diagram
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=multi
#SBATCH --output=logs/phase_diagram_%j.out
#SBATCH --error=logs/phase_diagram_%j.err

# Genera el diagrama de fases SoC_max(log Xi, log l) con el modelo del continuo
# (galpynostatic) -- la imagen "original" densa que despues consume el barrido DIP.
# Es rapido y no necesita GPU; corre igual en local sin SLURM:
#     ./venv/bin/python -m dip.phase_diagram
#
# Con RES se fija la resolucion y las rutas de salida acordes a slurm/dip_sweep.slurm.
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
mkdir -p logs data/restoration results_phase_diagram

export PYTHONUNBUFFERED=1
export NUM_XI="${NUM_XI:-$RES}"
export NUM_ELL="${NUM_ELL:-$RES}"
export OUT_PNG="./data/restoration/phase_diagram_sim_${RES}.png"
export OUT_REF_PNG="./results_phase_diagram/phase_diagram_reference_${RES}.png"
export OUT_NPY="./results_phase_diagram/phase_diagram_soc_${RES}.npy"

./venv/bin/python -m dip.phase_diagram
