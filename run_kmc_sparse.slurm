#!/bin/bash
#SBATCH --job-name=kmc_sparse
#SBATCH --partition=multi
#SBATCH --nodes=4
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/kmc_sparse_%j.out
#SBATCH --error=logs/kmc_sparse_%j.err

. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module load gcc/12.3.0

mkdir -p logs

# ============================================================
# Igual que run_kmc_param.slurm (mismo semaforo con named pipe),
# pero la lista de puntos (xi, el) no es un producto cartesiano:
# viene de un archivo generado por sample_sparse_points.py, que
# elige una muestra dispersa sobre la grilla del mapa de fases
# (para despues reconstruir el mapa completo con DIP).
# ============================================================
POINTS_FILE="kmc_sample_points.txt"
NRUNS=1          # trayectorias independientes por punto (subir si se necesita promediar)

EXEC="./KMC-Galvanostatic_noclus_param_claude"

SAFETY_MARGIN=3600   # 1 hora de margen
JOB_SECS=$((2*24*3600))
DEADLINE=$((SECONDS + JOB_SECS - SAFETY_MARGIN))

# ============================================================
# CONSTRUIR COLA DE TRABAJOS a partir del archivo de puntos
# ============================================================
declare -a queue
while read -r xi el; do
  [[ -z "$xi" ]] && continue
  for run in $(seq 1 $NRUNS); do
    queue+=("$xi $el $run")
  done
done < "$POINTS_FILE"

total=${#queue[@]}
NTASKS=$SLURM_NTASKS

echo "$(date): Iniciando. Total trabajos: $total | Ranuras: $NTASKS"
echo "$(date): Deadline en $((JOB_SECS - SAFETY_MARGIN)) segundos"

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

for i in $(seq 1 $NTASKS); do echo >&3; done

completed=0
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
    echo >&3
  ) &

  launched=$((launched + 1))
  echo "$(date): Lanzado [$launched/$total] xi=$xi el=$el run=$run (PID $!)"

done

wait
exec 3>&-

echo "$(date): Fin. Completados: $launched / $total"
