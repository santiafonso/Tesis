#!/bin/bash
#SBATCH --job-name=kmc_tanda
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=logs/tanda_%j.out
#SBATCH --error=logs/tanda_%j.err
# Una tanda del planificador (autoexp/kmc_planner.py next -> tanda_XX.csv) en un nodo de Serafin.
# 1 nucleo por corrida, NREP replicas por punto para llenar el nodo, y un archivo de estado por
# corrida con el codigo de salida (autoexp/kmc_results.py lo lee):
#
#   runs/r<row>_c<col>_k<rep>/datos-*.dat   salida del KMC
#   runs/r<row>_c<col>_k<rep>/estado.txt    "rc=corriendo inicio=<epoch>" mientras corre;
#                                           "rc=<n> segundos=<s>" al terminar
#                                           (0 = terminado, 124 = cortado por tope, otro = error)
#
# Uso (desde la carpeta de la tanda, que tiene tanda.csv y logs/):
#   sbatch ~/kmc/kmc_tanda.slurm [tanda.csv]                        Serafin: nodo entero (64 nucleos)
#   sbatch -p short --time=00:59:00 -c 10 ~/kmc/kmc_tanda.slurm      Mulatona: pocos nucleos (puntos rapidos)
# tanda.csv: row,col,logxi,logell,xi,ell[,g]. Con la columna g (opcional) cada punto corre con su g
# (4to argumento del KMC) en runs/g<g>_r..._c..._k.../: varios g en un mismo nodo. Sin g: -4.
# Env: KMC_DIR (default ~/kmc: kmc_fast + CS-40x20x40.xyz), NREP (default: nucleos / puntos),
#      MARGIN (segundos antes del fin del job en que se corta cada corrida, default 600),
#      TOPE_MAX (segundos maximos por corrida, p. ej. al correr como step dentro de un job largo),
#      KEEP_VMD=1 para no borrar los vmd-*.xyz (80-180 MB por corrida).
# Reenviar la misma tanda saltea las corridas con rc=0 (retoma cortadas y errores).
set -u
TANDA=${1:-tanda.csv}
KMC_DIR=${KMC_DIR:-$HOME/kmc}
MARGIN=${MARGIN:-600}
NCORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
EXE="$KMC_DIR/kmc_fast"
XYZ="$KMC_DIR/CS-40x20x40.xyz"
[[ -x $EXE && -f $XYZ && -f $TANDA ]] || { echo "falta $EXE, $XYZ o $TANDA"; exit 1; }

# 1 hilo por corrida y sin pinning de OpenMP (con OMP_PROC_BIND todas irian al mismo nucleo)
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
unset OMP_PLACES

# Fin del job: tiempo restante segun SLURM ([D-]HH:MM:SS, MM:SS)
left=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null)
to_secs() {
  local s=$1 d=0
  [[ $s == *-* ]] && { d=${s%%-*}; s=${s#*-}; }
  IFS=: read -ra p <<< "$s"
  local t=0; for x in "${p[@]}"; do t=$((t * 60 + 10#$x)); done
  echo $((d * 86400 + t))
}
if [[ -n $left && $left != INVALID && $left != UNLIMITED ]]; then END=$(( $(date +%s) + $(to_secs "$left") ))
else END=$(( $(date +%s) + 2 * 86400 )); fi

mapfile -t PTS < <(tail -n +2 "$TANDA" | grep -v '^\s*$')
NPTS=${#PTS[@]}
NREP=${NREP:-$(( NCORES / NPTS > 0 ? NCORES / NPTS : 1 ))}
echo "$(date): $NPTS puntos x $NREP replicas en $NCORES nucleos; fin del job $(date -d @$END)"

mkdir -p runs
launched=0
for rep in $(seq 1 "$NREP"); do
  for line in "${PTS[@]}"; do
    IFS=, read -r row col logxi logell _ _ g <<< "$line"
    g=${g//[[:space:]]/}
    d=$(printf "runs/%sr%03d_c%03d_k%d" "${g:+g${g}_}" "$row" "$col" "$rep")
    grep -q '^rc=0 ' "$d/estado.txt" 2>/dev/null && continue
    while (( $(jobs -rp | wc -l) >= NCORES )); do wait -n; done
    tope=$(( END - $(date +%s) - MARGIN ))
    [[ -n ${TOPE_MAX:-} ]] && (( tope > TOPE_MAX )) && tope=$TOPE_MAX
    (( tope > 60 )) || { echo "$(date): sin tiempo, no lanzo mas"; break 2; }
    mkdir -p "$d"; rm -f "$d"/datos-*.dat "$d"/vmd-*.xyz "$d"/parametros.dat
    ln -sf "$XYZ" "$d/CS-40x20x40.xyz"
    (
      cd "$d" || exit
      t0=$(date +%s)
      echo "rc=corriendo inicio=$t0" > estado.txt
      timeout "$tope" "$EXE" "$logxi" "$logell" "$rep" ${g:+"$g"} > kmc.out 2>&1
      rc=$?
      echo "rc=$rc segundos=$(( $(date +%s) - t0 )) xi=$logxi el=$logell g=${g:--4}" > estado.txt
      [[ ${KEEP_VMD:-0} == 1 ]] || rm -f vmd-*.xyz
    ) &
    launched=$((launched + 1))
    echo "$(date): [$launched] ${g:+g$g }r$row c$col k$rep  log xi $logxi  log l $logell  (tope ${tope}s)"
    sleep 1.1  # el KMC siembra con srand(time(0)): un arranque por segundo -> semillas distintas
  done
done
wait
echo "$(date): fin. estados:"; cat runs/*/estado.txt | cut -d' ' -f1 | sort | uniq -c
