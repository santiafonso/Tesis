#!/bin/bash
#SBATCH --job-name=kmc_full
#SBATCH --partition=multi
#SBATCH --nodes=4
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/kmc_full_%j.out
#SBATCH --error=logs/kmc_full_%j.err

. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module load gcc/12.3.0

mkdir -p logs

# ============================================================
# Corre los 205 puntos del sweep KMC disperso (kmc_sample_points.txt)
# desde cero, sin tope de tiempo por punto: cada uno corre hasta que
# el bucle principal (while (mui < muoff), linea ~290 del .cpp) llegue
# solo al cutoff de voltaje. El --gres=gpu:2 se mantiene -- es politica
# obligatoria de la particion "multi" (min 1 GPU/job, max 10 cores/GPU),
# no un cuello de botella real (los nodos ivb0x no tienen GPU fisica).
# Mismo semaforo con named pipe que run_kmc_param.slurm / run_kmc_sparse.slurm.
# ============================================================
POINTS_FILE="kmc_sample_points.txt"
NRUNS=1

EXEC="./KMC-Galvanostatic_noclus_param_claude"

SAFETY_MARGIN=3600
JOB_SECS=$((2*24*3600))
DEADLINE=$((SECONDS + JOB_SECS - SAFETY_MARGIN))

declare -a queue
while read -r xi el; do
  [[ -z "$xi" ]] && continue
  for run in $(seq 1 $NRUNS); do
    queue+=("$xi $el $run")
  done
done < "$POINTS_FILE"

total=${#queue[@]}
NTASKS=$SLURM_NTASKS

echo "$(date): Iniciando. Total trabajos: $total | Ranuras: $NTASKS | Sin tope por punto"
echo "$(date): Deadline en $((JOB_SECS - SAFETY_MARGIN)) segundos"

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

for i in $(seq 1 $NTASKS); do echo >&3; done

launched=0

for item in "${queue[@]}"; do

  if [[ $SECONDS -ge $DEADLINE ]]; then
    echo "$(date): Tiempo limite alcanzado. Lanzados: $launched / $total"
    break
  fi

  read -u3

  if [[ $SECONDS -ge $DEADLINE ]]; then
    echo "$(date): Tiempo limite alcanzado tras espera. Lanzados: $launched / $total"
    echo >&3
    break
  fi

  read -r xi el run <<< "$item"
  (
    srun --exclusive -N1 -n1 -c"$SLURM_CPUS_PER_TASK" \
         "$EXEC" "$xi" "$el" "$run"
    echo "$(date): Terminado xi=$xi el=$el run=$run"
    echo >&3
  ) &

  launched=$((launched + 1))
  echo "$(date): Lanzado [$launched/$total] xi=$xi el=$el run=$run (PID $!)"

done

wait
exec 3>&-

echo "$(date): Fin. Lanzados: $launched / $total"
