#!/bin/bash
# Corre los puntos de tanda.csv con kmc_ef_rand (superficie + CalcularEf), 4 a la vez, 1 hilo c/u.
cd "$(dirname "$0")"
tail -n +2 tanda.csv | awk -F, '{print NR-1, $1, $2, $3, $4, $7}' | \
  xargs -P 4 -L 1 bash -c 'i=$0; r=$1; c=$2; lx=$3; ll=$4; g=$5
    d=r$(printf %03d $r)_c$(printf %03d $c); mkdir -p $d; ln -sf ../../CS-40x20x40.xyz $d/
    sleep $((i % 4 * 2))
    cd $d && OMP_NUM_THREADS=1 nice -n 10 ../../kmc_ef_rand $lx $ll 1 $g > kmc.out 2>&1; echo rc=$? > estado.txt'
echo "terminado $(date)" > TERMINADO
