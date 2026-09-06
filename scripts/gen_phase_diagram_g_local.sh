#!/bin/bash
# Genera los mapas de continuo del barrido de g EN LOCAL (el cluster no tiene
# Python moderno para galpynostatic). ~71 min por mapa a 128x128; corre de a 2
# en paralelo (NTHREADS=2, la maquina tiene 4 cores). Idempotente: salta el g
# cuyo soc.npy ya existe.
#
#   ./scripts/gen_phase_diagram_g_local.sh            # 128x128 (default)
#   RES=64 PAR=2 ./scripts/gen_phase_diagram_g_local.sh
set -u
cd "$(dirname "$0")/.." || exit 1

RES="${RES:-128}"
PAR="${PAR:-2}"
NTHREADS_EACH="${NTHREADS_EACH:-2}"
GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 2.0 4.0)

run_one() {
    local g="$1"
    local d="results/phase_diagram_g/g${g}"
    mkdir -p "$d"
    if [ -f "$d/soc.npy" ]; then
        echo "[$(date +%H:%M:%S)] g=$g ya hecho, salto"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] g=$g START"
    NUM_XI="$RES" NUM_ELL="$RES" NTHREADS="$NTHREADS_EACH" G="$g" \
        OUT_PNG="$d/sim_${RES}.png" OUT_REF_PNG="$d/reference.png" OUT_NPY="$d/soc.npy" \
        ./venv/bin/python -m dip.phase_diagram > "$d/run.log" 2>&1
    echo "[$(date +%H:%M:%S)] g=$g DONE (exit $?)"
}
export -f run_one
export RES NTHREADS_EACH

printf '%s\n' "${GS[@]}" | xargs -P "$PAR" -I {} bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] barrido de g completo -> results/phase_diagram_g/"
