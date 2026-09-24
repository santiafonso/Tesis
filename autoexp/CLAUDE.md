# autoexp/ — loop de experimentación automática (branch `autoexp`)

Objetivo (23/9): reconstruir el diagrama de fases 128×128 con **≤ 64 puntos consultados** y
**> 38 dB** de PSNR. Si no se llega, lo más alto posible. Métrica de avance: `mean_psnr` en los
`g` de desarrollo (-4.0, -2.0, -0.5), siempre reportada junto a `min_psnr` (el peor `g`).
Validación final de lo que gane: los 9 `g` negativos (`--gs all`).

## Archivos FIJOS (el loop no los modifica nunca)

- `oracle.py`: única vía para ver valores de la imagen. Cuenta y registra las consultas.
- `eval.py`: calcula el puntaje, verifica el presupuesto y que el registro sea real, y escribe
  `score.json`, `panel.png` y `leaderboard.csv`.
- `results/phase_diagram_g/*/sim_128.png`: la verdad.

## Archivos modificables

`sampling.py`, `interp.py`, `trial.py`, `optuna_worker.py`, `slurm/autoexp_*.slurm`, y las
perillas de `dip/restoration.py` (con los defaults históricos intactos).

## Reglas

1. **Ningún método mira la imagen real.** Solo `oracle.query()` y `oracle.observed()`. Los
   pseudo-puntos (valores inventados a partir de lo consultado) están permitidos y no gastan
   presupuesto. Una máscara tipo `frontier_mix`, que usa el campo denso, solo vale como cota de
   referencia y se anota como "oráculo" en la bitácora; nunca cuenta como resultado.
2. Un cambio por intento. Se corre `python -m autoexp.trial <nombre> '<params>'`, o el eval si
   la reconstrucción se hizo por otra vía, y se anota en `experimentos.md`: qué se probó, el
   puntaje y si queda o se descarta.
3. Si un cambio de código empeora el puntaje, se revierte con git. Los parámetros van en el
   nombre o la nota de la corrida, no en el código.
4. DIP corre en Mendieta (A30: ~12 min cada 16k iteraciones), nunca local (0.7 s/it en CPU).
   Interpolación y muestreo sí corren local, en segundos.
5. Todo lo que salga bien tiene que dejar una figura lista para diapositiva (`panel.png`).

## Cómo correr

```
# local, sin DIP (segundos)
./venv/bin/python -m autoexp.trial adapt48_tps '{"sampler":"adaptive","sampler_kw":{"n1":48},"recon":"rbf_tps"}' --note "..."
# cluster: estudio Optuna (4 workers en un mismo estudio)
sbatch slurm/autoexp_optuna.slurm          # desde ~/Tesis-autoexp en Mendieta
```

Clon de trabajo: local `~/Tesis-autoexp` (worktree del branch `autoexp`), y en Mendieta
`~/Tesis-autoexp` (worktree de `~/Tesis-frontier`, que tiene cambios locales propios: no
tocarlo). En los dos, `venv` y `results/phase_diagram_g` son symlinks al clon principal. En el
cluster, el worker usa `~/venv-autoexp` (py3.11 + optuna) y DIP usa `./venv` (py3.6).
