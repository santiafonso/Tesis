#!/bin/bash
#SBATCH --job-name=kmc_param
#SBATCH --partition=multi
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --time=02-00:00:00

. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module load gcc/12.3.0

# ============================================================
# LISTAS DE PARAMETROS — editar segun necesidad
# ============================================================
xi_list=("-1.5" "-1.0" "-0.5" "0.0" "0.5" "1.0")
el_list=("-1.147" "-0.8" "-0.5" "-0.2")
NRUNS=4          # trayectorias independientes por combinacion (xi, el)

EXEC="./KMC-Galvanostatic_noclus_param_claude"  

# ============================================================
# TIEMPO LIMITE
# Parar SAFETY_MARGIN segundos antes del limite de la particion
# para no dejar jobs cortados a la mitad
# ============================================================
SAFETY_MARGIN=3600   # 1 hora de margen
JOB_SECS=$((2*24*3600))
DEADLINE=$((SECONDS + JOB_SECS - SAFETY_MARGIN))

# ============================================================
# CONSTRUIR COLA DE TRABAJOS
# Orden: xi externo, el medio, run interno
# ============================================================
declare -a queue
for xi in "${xi_list[@]}"; do
  for el in "${el_list[@]}"; do
    for run in $(seq 1 $NRUNS); do
      queue+=("$xi $el $run")
    done
  done
done
total=${#queue[@]}
NTASKS=$SLURM_NTASKS   # ranuras simultaneas (= --ntasks del SBATCH)

echo "$(date): Iniciando. Total trabajos: $total | Ranuras: $NTASKS"
echo "$(date): Deadline en $((JOB_SECS - SAFETY_MARGIN)) segundos"

# ============================================================
# SEMAFORO CON NAMED PIPE
# Garantiza que siempre haya exactamente NTASKS jobs corriendo.
# Cuando un job termina libera su ranura y el siguiente arranca
# de inmediato, sin esperar al resto del batch.
# ============================================================
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"          # desvincula del filesystem; fd 3 sigue abierto

# Cargar NTASKS tokens iniciales (una por ranura disponible)
for i in $(seq 1 $NTASKS); do echo >&3; done

completed=0
launched=0

for item in "${queue[@]}"; do

  # Verificar tiempo antes de consumir una ranura
  if [[ $SECONDS -ge $DEADLINE ]]; then
    echo "$(date): Tiempo limite alcanzado. Lanzados: $launched / $total"
    break
  fi

  # Esperar una ranura libre (bloquea si todas ocupadas)
  read -u3

  # Relanzar verificacion de tiempo tras esperar
  if [[ $SECONDS -ge $DEADLINE ]]; then
    echo "$(date): Tiempo limite alcanzado tras espera. Lanzados: $launched / $total"
    echo >&3   # devolver token para no bloquear el wait final
    break
  fi

  # Lanzar job en subshell; al terminar devuelve la ranura
  read -r xi el run <<< "$item"
  (
    srun --exclusive -N1 -n1 -c"$SLURM_CPUS_PER_TASK" \
         "$EXEC" "$xi" "$el" "$run"
    echo >&3   # devolver ranura al pipe
  ) &

  launched=$((launched + 1))
  echo "$(date): Lanzado [$launched/$total] xi=$xi el=$el run=$run (PID $!)"

done

# Esperar a que todos los jobs en vuelo terminen
wait
exec 3>&-

echo "$(date): Fin. Completados: $launched / $total"
