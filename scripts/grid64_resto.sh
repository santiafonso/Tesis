#!/bin/bash
# Resto de la grilla KMC 64x64 de g=-4 (8/10): columnas log l < -3 + los 40 cortados de log l ~ -2.9,
# con tope de 24 h por corrida (TOPE_MAX). Cuando arranca el contenedor k, corre ahi la parte k
# (~/kmc/tandas/g-4_grid64_resto_p<k>/, nice 19 como la grilla). Corre en el login de Mendieta:
#   setsid nohup scripts/grid64_resto.sh "431233 431234 431235 431236" >> autoexp/kmc/g-4/grid64/resto/lanzador.log 2>&1 &
SRF=siaosorio@serafin.ccad.unc.edu.ar
read -ra CONT <<< "$1"
declare -A done_k
while (( ${#done_k[@]} < ${#CONT[@]} )); do
  for k in "${!CONT[@]}"; do
    [[ -n ${done_k[$k]:-} ]] && continue
    j=${CONT[$k]}; p=$((k + 1)); d="~/kmc/tandas/g-4_grid64_resto_p$p"
    st=$(ssh -o BatchMode=yes $SRF "squeue -h -j $j -o %T" 2>/dev/null)
    if [[ $st == RUNNING ]]; then
      ssh -o BatchMode=yes $SRF "cd $d && nohup setsid srun --jobid=$j --overlap -N1 -n1 -c64 env TOPE_MAX=86400 nice -n 19 bash ~/kmc/kmc_tanda.slurm tanda.csv > logs/step.out 2> logs/step.err < /dev/null &" & pid=$!
      sleep 30; kill $pid 2>/dev/null  # ssh no vuelve mientras srun vive (ver kmc_auto.run_step)
      echo "$(date): parte $p lanzada en $j"; done_k[$k]=1
    fi
  done
  (( ${#done_k[@]} < ${#CONT[@]} )) && sleep 600
done
echo "$(date): todas las partes lanzadas"
