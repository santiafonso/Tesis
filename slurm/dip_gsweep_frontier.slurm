#!/bin/bash
#SBATCH --job-name=dip_gsweep_fr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --array=0-224%12
#SBATCH --output=logs/dip_gsweep_fr_%A_%a.out
#SBATCH --error=logs/dip_gsweep_fr_%A_%a.err

# Reconstruccion DIP del barrido de g atacando el problema de los g negativos
# (frente de fase abrupto -> el DIP lo emborrona y el backtracking oscila):
#   1. mascara de FRONTERA (frontier_mix, MIX_FRAC=0.40) en vez de Bernoulli
#      uniforme -> los puntos observados caen sobre el frente pero las mesetas
#      conservan ~40 % de los puntos (si no, sus niveles derivan)
#   2. mas iteraciones (NUM_ITER=16000)   -> el borde converge tarde
#   3. menos ruido de regularizacion (REG_NOISE_STD=0.01) -> deja fijar el escalon
#   4. backtracking robusto: PSNR_DROP_TOL=-8 (vs. media movil) + MAX_FALLBACKS=3
#      -> no entra en bucle de rollback cuando el PSNR_masked oscila sobre el frente
# Se incluye la familia `uniform` al MISMO N y MISMOS hiperparametros como
# baseline apples-to-apples (no comparar contra results/dip_gsweep/ ni contra la
# tanda v1, que corrieron con parametros distintos).
# Reunion 9/9, punto 3: sumadas `frontier` (solo frontera), `spread` (evita la
# frontera) y `grid` (grilla 2D regular, sin azar) para comparar las 5
# distribuciones de muestreo en el regimen de g negativo (el caso real, punto 5).
#
#   9 g (negativos + 0.0) x 5 mascaras x 5 MASK_FRAC = 225 tareas; %12 simultaneas.
#   Idempotente: las 90 tareas frontier_mix/uniform ya corridas se saltan solas.
#
#   lee  data/restoration/masks_g/g<g>/mf<frac>/mask_<name>.npy
#        (generar antes con slurm/frontier_mask_g.slurm)
#        results/phase_diagram_g/g<g>/sim_<RES>.png  (fallback data/)
#   escribe results/dip_gsweep_frontier/g<g>/mf<frac>/<name>/
#
# Idempotente: salta si .../restored.npy existe. Borrar la carpeta para reforzar.
#
# Submit:  sbatch slurm/dip_gsweep_frontier.slurm
#          sbatch --dependency=afterok:<jobid_frontier_mask_g> slurm/dip_gsweep_frontier.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0)
MASKS=(frontier frontier_mix spread uniform grid)
MFS=(0.900 0.950 0.980 0.990 0.995)

NM=${#MASKS[@]}
NF=${#MFS[@]}
PER_G=$((NM * NF))

i=${SLURM_ARRAY_TASK_ID:-0}
g=${GS[$((i / PER_G))]}
r=$((i % PER_G))
name=${MASKS[$((r / NF))]}
mf=${MFS[$((r % NF))]}

export PYTHONUNBUFFERED=1
IMAGE_PATH="results/phase_diagram_g/g${g}/sim_${RES}.png"
[ -f "$IMAGE_PATH" ] || IMAGE_PATH="data/restoration/phase_diagram_g/g${g}/sim_${RES}.png"
export IMAGE_PATH
export MASK_PATH="data/restoration/masks_g/g${g}/mf${mf}/mask_${name}.npy"
export NUM_ITER="${NUM_ITER:-16000}"
export REG_NOISE_STD="${REG_NOISE_STD:-0.01}"
export PSNR_DROP_TOL="${PSNR_DROP_TOL:--8}"
export MAX_FALLBACKS="${MAX_FALLBACKS:-3}"
export SHOW_EVERY="${SHOW_EVERY:-100}"
export SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="results/dip_gsweep_frontier/g${g}/mf${mf}/${name}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "!! falta $IMAGE_PATH  (correr antes slurm/phase_diagram_g.slurm)"
    exit 1
fi
if [ ! -f "$MASK_PATH" ]; then
    echo "!! falta $MASK_PATH  (correr antes slurm/frontier_mask_g.slurm)"
    exit 1
fi
if [ -f "$OUTPUT_DIR/restored.npy" ]; then
    echo "== ya hecho: $OUTPUT_DIR =="
    exit 0
fi

echo "=== task $i  ->  g=$g  mask=$name  MASK_FRAC=$mf  ($(date)) ==="
./venv/bin/python -m dip.restoration
echo "=== done task $i ($(date)) ==="
