#!/bin/bash
#SBATCH --job-name=kmc_dip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=logs/kmc_dip_%j.out
#SBATCH --error=logs/kmc_dip_%j.err
# DIP sobre los puntos KMC preparados con `python -m autoexp.kmc_dip prepare` (Mendieta):
# una corrida por semilla (DIPDIR/s*/gkmc/, con env.json), en paralelo en la misma GPU.
#   sbatch --export=ALL,DIPDIR=autoexp/kmc/g-4/dip slurm/kmc_dip.slurm
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1
PY="${DIP_PY:-$PWD/venv/bin/python}"
pids=()
for g in "$DIPDIR"/s*/gkmc; do
  (
    eval "$("$PY" -c "import json,shlex;[print('export %s=%s' % (k, shlex.quote(v))) for k, v in json.load(open('$g/env.json')).items()]")"
    export IMAGE_PATH="$g/target.png" MASK_PATH="$g/mask.npy" OUTPUT_DIR="$g/dip"
    "$PY" -m dip.restoration > "$g/dip.log" 2>&1 && cp "$g/dip/restored.npy" "$g/restored.npy"
  ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "fin rc=$rc $(date)"; exit $rc
