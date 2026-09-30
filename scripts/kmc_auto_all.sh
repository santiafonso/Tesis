#!/bin/bash
# Lanza (o relanza) un loop autoexp.kmc_auto por cada g con estado en autoexp/kmc/g*/kmc.json que no
# haya terminado (terminado = existe control.json) y que no este corriendo ya. Lo usa el crontab
# @reboot; se puede correr a mano cuantas veces se quiera.
cd "$(dirname "$0")/.." || exit 1
for st in autoexp/kmc/g*/kmc.json; do
  d=$(dirname "$st")
  [[ -f $d/control.json ]] && continue
  pgrep -f -- "autoexp.kmc_auto --state $st\$" > /dev/null && continue
  setsid nohup ./venv/bin/python -m autoexp.kmc_auto --state "$st" >> "$d/auto.log" 2>&1 < /dev/null &
  echo "lanzado $st"
  sleep 20  # no abrir todas las conexiones ssh a la vez (el login de CCAD corta con "Connection reset")
done
