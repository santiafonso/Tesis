#!/bin/bash
#SBATCH --job-name=kmc_test
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/kmc_test_%j.out
#SBATCH --error=logs/kmc_test_%j.err

. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module load gcc/11.2.0

mkdir -p logs

# ============================================================
# PRUEBA MINIMA — 2 puntos (xi, el), 1 corrida cada uno.
# Objetivo: validar que el pipeline completo (compilacion, lectura
# de CS-40x20x40.xyz, escritura de datos-*.dat) funciona en el
# cluster antes de lanzar el barrido completo (run_kmc_param.slurm).
# ============================================================
xi_list=("-1.0" "0.0")
el_list=("-1.147" "-1.147")
NRUNS=1

EXEC="./KMC-Galvanostatic_noclus_param_claude"

echo "$(date): Prueba iniciada. NTASKS=$SLURM_NTASKS"

for idx in "${!xi_list[@]}"; do
  xi="${xi_list[$idx]}"
  el="${el_list[$idx]}"
  for run in $(seq 1 $NRUNS); do
    srun --exclusive -N1 -n1 -c"$SLURM_CPUS_PER_TASK" \
         "$EXEC" "$xi" "$el" "$run" &
    echo "$(date): Lanzado xi=$xi el=$el run=$run"
  done
done

wait
echo "$(date): Prueba terminada."
