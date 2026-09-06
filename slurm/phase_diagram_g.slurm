#!/bin/bash
#SBATCH --job-name=phase_diagram_g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=multi
#SBATCH --array=0-10%11
#SBATCH --output=logs/phase_diagram_g_%A_%a.out
#SBATCH --error=logs/phase_diagram_g_%A_%a.err

# Barrido del parametro de interaccion g (isoterma de Frumkin) del modelo del
# continuo: un mapa SoC_max(log Xi, log l) por valor de g, una tarea por g.
# El profe marco que "los mas conflictivos son los negativos", asi que el barrido
# carga sobre todo g<0 (mas g=0 Langmuir y g=2,4 repulsivos de contraste).
#
# Salida por g (en results/, git-ignored; se promueve a data/ el/los g que se
# terminen usando como input DIP):
#   results/phase_diagram_g/g<val>/sim_<RES>.png   (input DIP, gris crudo)
#   results/phase_diagram_g/g<val>/reference.png   (viridis + ejes)
#   results/phase_diagram_g/g<val>/soc.npy (+ _axes.npz)
#
# Idempotente: si soc.npy ya existe, la tarea se saltea.
#
# Submit:  sbatch slurm/phase_diagram_g.slurm
#          RES=64 sbatch --export=ALL slurm/phase_diagram_g.slurm   # mas rapido
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

RES="${RES:-128}"
GS=(-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 2.0 4.0)
g=${GS[${SLURM_ARRAY_TASK_ID:-0}]}

OUT_DIR="results/phase_diagram_g/g${g}"
mkdir -p logs "$OUT_DIR"

export PYTHONUNBUFFERED=1
export NUM_XI="${NUM_XI:-$RES}"
export NUM_ELL="${NUM_ELL:-$RES}"
export G="$g"
export OUT_PNG="${OUT_DIR}/sim_${RES}.png"
export OUT_REF_PNG="${OUT_DIR}/reference.png"
export OUT_NPY="${OUT_DIR}/soc.npy"

if [ -f "$OUT_NPY" ]; then
    echo "== ya hecho: $OUT_NPY (borrar para reforzar) =="
    exit 0
fi

echo "=== g=$g  RES=$RES  ($(date)) ==="
./venv/bin/python -m dip.phase_diagram
echo "=== done g=$g ($(date)) ==="
