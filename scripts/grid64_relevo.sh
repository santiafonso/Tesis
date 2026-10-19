#!/bin/bash
# Relevo de la grilla KMC 64x64 de g=-4 (autoexp/kmc_grid.py) entre contenedores de Serafin: cuando
# arranca el contenedor nuevo de la parte k, relanza ahi la parte (nice 19, kmc_tanda.slurm saltea lo
# que ya tiene rc=0). Corre en el login de Mendieta:
#   setsid nohup scripts/grid64_relevo.sh "430773 430774 430775 430776" >> autoexp/kmc/g-4/grid64/relevo.log 2>&1 &
# Argumento: los contenedores nuevos, en el orden de las partes 1..4.
SRF=siaosorio@serafin.ccad.unc.edu.ar
read -ra NEW <<< "$1"
declare -A done_k
while (( ${#done_k[@]} < ${#NEW[@]} )); do
  for k in "${!NEW[@]}"; do
    [[ -n ${done_k[$k]:-} ]] && continue
    j=${NEW[$k]}; p=$((k + 1)); d="~/kmc/tandas/g-4_grid64_p$p"
    if ssh -o BatchMode=yes $SRF "grep -q ' fin. estados' $d/logs/step.out 2>/dev/null && ! grep -q 'sin tiempo' $d/logs/step.out"; then
      echo "$(date): parte $p ya termino, no se relanza"; done_k[$k]=1; continue
    fi
    st=$(ssh -o BatchMode=yes $SRF "squeue -h -j $j -o %T" 2>/dev/null)
    if [[ $st == RUNNING ]]; then
      ssh -o BatchMode=yes $SRF "cd $d && mv logs/step.out logs/step_\$(date +%s).out 2>/dev/null; nohup setsid srun --jobid=$j --overlap -N1 -n1 -c64 nice -n 19 bash ~/kmc/kmc_tanda.slurm tanda.csv > logs/step.out 2> logs/step.err < /dev/null &" & pid=$!
      sleep 30; kill $pid 2>/dev/null  # ssh no vuelve mientras srun vive (ver kmc_auto.run_step)
      echo "$(date): parte $p relanzada en $j"; done_k[$k]=1
    fi
  done
  sleep 600
done
echo "$(date): relevo completo"
