#!/bin/bash
#SBATCH --job-name=kmc_demo
#SBATCH --partition=multi
#SBATCH --nodes=2
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/kmc_demo_%j.out
#SBATCH --error=logs/kmc_demo_%j.err

. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module load gcc/12.3.0

mkdir -p logs

# ============================================================
# Version demo de run_kmc_sparse_v2.slurm: corre solo un subconjunto
# chico (60 puntos nuevos, kmc_sample_points_demo.txt) con footprint
# reducido (2 nodos/10 slots) y tope de 12 min por punto, para tener
# un lote de datos KMC reales rapido y poder correr DIP sobre eso.
# Peor caso: (60/10)*12min = 72 min.
# ============================================================
POINTS_FILE="kmc_sample_points_demo.txt"
NRUNS=1
POINT_TIMEOUT="12m"

EXEC="./KMC-Galvanostatic_noclus_param_claude"

SAFETY_MARGIN=600
JOB_SECS=$((2*3600))
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

echo "$(date): Iniciando. Total trabajos: $total | Ranuras: $NTASKS | Tope por punto: $POINT_TIMEOUT"

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

  read -r xi el run <<< "$item"
  (
    if timeout "$POINT_TIMEOUT" srun --exclusive -N1 -n1 -c"$SLURM_CPUS_PER_TASK" \
         "$EXEC" "$xi" "$el" "$run"; then
      echo "$(date): OK xi=$xi el=$el run=$run"
    else
      echo "$(date): TIMEOUT/ERROR xi=$xi el=$el run=$run"
    fi
    echo >&3
  ) &

  launched=$((launched + 1))
  echo "$(date): Lanzado [$launched/$total] xi=$xi el=$el run=$run (PID $!)"

done

wait
exec 3>&-

echo "$(date): Fin. Lanzados: $launched / $total"
