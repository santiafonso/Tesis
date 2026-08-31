#!/bin/bash
#SBATCH --job-name=kmc_sparse_v2
#SBATCH --partition=multi
#SBATCH --nodes=4
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=18:00:00
#SBATCH --output=logs/kmc_sparse_v2_%j.out
#SBATCH --error=logs/kmc_sparse_v2_%j.err

. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module load gcc/12.3.0

mkdir -p logs

# ============================================================
# v2: correccion sobre run_kmc_sparse.slurm (job 1167939, que en
# 2 dias solo termino 4/205 puntos):
#
# (el --gres=gpu:2 original SE MANTIENE: sacct muestra que la
# particion "multi" exige gres/gpu>=1 por job y max 10 cores/GPU
# -- con 20 cpus/nodo hacen falta 2 GPUs/nodo por politica, aunque
# los nodos ivb0x no tienen GPU fisica en sinfo (gres de facturacion
# del sitio, no cuello de botella real). El job 1167939 ya corria
# con gres/gpu=8 sin tiempo de cola raro; el timeout fue por
# variabilidad real de tiempo por punto, no por esperar GPU.)
#
# Cada punto ahora corre con un `timeout` (POINT_TIMEOUT). El bucle
#    principal del kMC (`while (mui < muoff)`, linea ~290 del .cpp) no
#    tiene limite de pasos: corre hasta que el potencial llega al cutoff,
#    y eso tardo entre 1.6 min y 6.7 h en los puntos que si terminaron la
#    vez pasada (medido con los mtimes de los .dat vs. el momento de
#    lanzamiento). Sin tope, unos pocos puntos lentos (cerca de la
#    frontera de fase, donde hacen falta muchos mas eventos KMC) acaparan
#    un slot del semaforo por horas y frenan el avance de los otros 205.
#    Con 20 slots en paralelo y un tope de POINT_TIMEOUT por punto, el
#    peor caso para terminar los 205 puntos es (205/20)*POINT_TIMEOUT,
#    bien adentro de las 18h pedidas.
# ============================================================
POINTS_FILE="kmc_sample_points.txt"
NRUNS=1                # trayectorias independientes por punto
POINT_TIMEOUT="90m"    # tope de tiempo real por punto (kill -TERM si se pasa)

EXEC="./KMC-Galvanostatic_noclus_param_claude"

SAFETY_MARGIN=1800   # 30 min de margen
JOB_SECS=$((18*3600))
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

echo "$(date): Iniciando. Total trabajos: $total | Ranuras: $NTASKS | Tope por punto: $POINT_TIMEOUT"
echo "$(date): Deadline en $((JOB_SECS - SAFETY_MARGIN)) segundos"

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

for i in $(seq 1 $NTASKS); do echo >&3; done

completed=0
launched=0
timed_out=0

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
