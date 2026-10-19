#!/bin/bash
# Lleva una tanda de corridas KMC largas (slurm/kmc/kmc_largo.sh, con checkpoint) de contenedor en
# contenedor hasta que terminan todas. Corre en el login de Mendieta:
#   setsid nohup scripts/kmc_largo_relevo.sh <carpeta en Serafin> <nucleos> "<cont1> <cont2> ..." \
#       >> <log> 2>&1 < /dev/null &
# Para cada contenedor, en orden: espera a que corra, lanza la tanda como step (srun --overlap) y
# espera a que el contenedor termine; ahi pasa al siguiente (cada corrida retoma de su checkpoint).
# Si se acaban los contenedores y faltan corridas, lo avisa y sale: relanzarlo con los IDs nuevos.
# NREP (env, default 2): replicas por punto, se pasa a kmc_largo.sh (la grilla 64x64 usa NREP=1).
SRF=siaosorio@serafin.ccad.unc.edu.ar
DIR=$1; NC=$2; read -ra CONT <<< "$3"; NREP=${NREP:-2}
q() { ssh -o BatchMode=yes $SRF "$@" 2>/dev/null; }
pendientes() { q "cd $DIR && n=\$(( \$(tail -n +2 tanda.csv | grep -c .) * $NREP )); t=\$(cat runs/*/estado.txt 2>/dev/null | grep -c '^rc=0 '); echo \$((n - t))"; }
for j in "${CONT[@]}"; do
  while :; do
    p=$(pendientes); [[ $p == 0 ]] && { echo "$(date): todas terminadas"; exit 0; }
    st=$(q "squeue -h -j $j -o %T")
    [[ -z $st ]] && st=$(q "sacct -n -X -j $j -o State | head -1 | tr -d ' '")  # vacio = red caida
    [[ $st == RUNNING ]] && break
    [[ $st == COMPLETED || $st == CANCELLED* || $st == FAILED || $st == TIMEOUT ]] && { echo "$(date): $j ya no esta ($st), sigo"; continue 2; }
    sleep 600
  done
  q "cd $DIR && mkdir -p logs && nohup setsid srun --jobid=$j --overlap -N1 -n1 -c$NC --export=ALL,NREP=$NREP bash ~/kmc/kmc_largo.sh tanda.csv >> logs/step.out 2>> logs/step.err < /dev/null &" & pid=$!
  sleep 30; kill $pid 2>/dev/null  # ssh no vuelve mientras srun vive (ver kmc_auto.run_step)
  echo "$(date): tramo lanzado en $j ($p corridas pendientes)"
  while [[ $(q "squeue -h -j $j -o %T") == RUNNING ]]; do
    [[ $(pendientes) == 0 ]] && { echo "$(date): todas terminadas"; exit 0; }
    sleep 600
  done
  echo "$(date): termino $j"
done
p=$(pendientes)
[[ $p == 0 ]] && echo "$(date): todas terminadas" || echo "$(date): faltan $p corridas y no hay mas contenedores: relanzar con IDs nuevos"
