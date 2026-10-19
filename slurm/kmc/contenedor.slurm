#!/bin/bash
#SBATCH --job-name=kmc_cont
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/cont_%j.out
# Job "contenedor": reserva un nodo entero 48 h sin hacer nada; los loops autoexp.kmc_auto
# (--reuse-job <id>) corren sus tandas adentro como steps (srun --overlap): la cola se paga una vez.
sleep 172800
