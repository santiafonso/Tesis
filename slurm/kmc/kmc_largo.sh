#!/bin/bash
# Corridas KMC largas, sin tope total: el KMC con checkpoint (kmc_ckpt, KMC_CKPT=1) corre hasta el fin
# del job que lo contiene (menos MARGIN), guarda el estado al recibir SIGTERM y la proxima vez que se
# lance esta misma tanda retoma cada corrida donde quedo (misma trayectoria bit a bit que sin cortar).
# Pensado para correr como step dentro de un contenedor (scripts/kmc_largo_relevo.sh lo relanza en el
# siguiente cuando vence el actual):
#   srun --jobid=<cont> --overlap -N1 -n1 -c<n> bash ~/kmc/kmc_largo.sh tanda.csv
#
# tanda.csv: row,col,logxi,logell,xi,ell,g (como kmc_tanda.slurm). runs/g<g>_r<row>_c<col>_k<rep>/:
#   datos-*.dat, ckpt-*.bin (mientras no termina) y estado.txt:
#     rc=corriendo inicio=<epoch> tramo=<n>
#     rc=0   segundos=<total>  -> terminado (kmc_results: SoC final)
#     rc=124 segundos=<total>  -> pausado con checkpoint (kmc_results: cota); relanzar retoma
# Env: KMC_DIR (default ~/kmc: kmc_ckpt + CS-40x20x40.xyz), NREP (default 2), MARGIN (default 900 s).
set -u
TANDA=${1:-tanda.csv}
KMC_DIR=${KMC_DIR:-$HOME/kmc}
MARGIN=${MARGIN:-900}
NREP=${NREP:-2}
EXE="$KMC_DIR/kmc_ckpt"
XYZ="$KMC_DIR/CS-40x20x40.xyz"
[[ -x $EXE && -f $XYZ && -f $TANDA ]] || { echo "falta $EXE, $XYZ o $TANDA"; exit 1; }
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false KMC_CKPT=1
unset OMP_PLACES

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
echo "$(date): ${#PTS[@]} puntos x $NREP replicas; fin del job $(date -d @$END)"
mkdir -p runs
for rep in $(seq 1 "$NREP"); do
  for line in "${PTS[@]}"; do
    IFS=, read -r row col logxi logell _ _ g <<< "$line"
    g=${g//[[:space:]]/}; g=${g:--4}
    d=$(printf "runs/g%s_r%03d_c%03d_k%d" "$g" "$row" "$col" "$rep")
    grep -q '^rc=0 ' "$d/estado.txt" 2>/dev/null && continue
    tope=$(( END - $(date +%s) - MARGIN ))
    (( tope > 300 )) || { echo "$(date): sin tiempo, no lanzo mas"; break 2; }
    mkdir -p "$d"
    ln -sf "$XYZ" "$d/CS-40x20x40.xyz"
    (
      cd "$d" || exit
      prev=0; tramo=1
      if ls ckpt-*.bin > /dev/null 2>&1; then
        prev=$(sed -n 's/.*segundos=\([0-9]*\).*/\1/p' estado.txt 2>/dev/null); prev=${prev:-0}
        t=$(sed -n 's/.*tramo=\([0-9]*\).*/\1/p' estado.txt 2>/dev/null); tramo=$(( ${t:-0} + 1 ))
      else
        rm -f datos-*.dat vmd-*.xyz parametros.dat  # arranque de cero (o el tramo anterior no llego a guardar)
      fi
      t0=$(date +%s)
      echo "rc=corriendo inicio=$t0 segundos=$prev tramo=$tramo" > estado.txt
      timeout "$tope" "$EXE" "$logxi" "$logell" "$rep" "$g" >> kmc.out 2>&1
      rc=$?
      echo "rc=$rc segundos=$(( prev + $(date +%s) - t0 )) tramo=$tramo xi=$logxi el=$logell g=$g" > estado.txt
      rm -f vmd-*.xyz
    ) &
    echo "$(date): g$g r$row c$col k$rep  log xi $logxi  log l $logell  (tope ${tope}s)"
    sleep 1.1  # srand(time(0)) en el arranque de cero: un segundo entre corridas -> semillas distintas
  done
done
wait
echo "$(date): fin del tramo. estados:"; cat runs/*/estado.txt | cut -d' ' -f1 | sort | uniq -c
