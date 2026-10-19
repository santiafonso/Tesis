# autoexp/kmc/ — qué hay en cada carpeta

Espejo local del estado del KMC real. **El estado vivo está en Mendieta** (`~/Tesis-autoexp/autoexp/kmc/` y
`autoexp/kmc_ef/`); acá se trae con `rsync`. Por eso las carpetas no se renombran: el código
(`kmc_auto`, `kmc_planner`, scripts de relevo) y los loops de Mendieta usan estas rutas. Historia completa en
`autoexp/CLAUDE.md`.

## Reconstrucciones por g (binario viejo, con los errores 1 y 3)

`g-8/` … `g4/` — una carpeta por g, 96 puntos cada una:
- `kmc.json` — estado del planificador (puntos medidos, pseudo, tandas, jobs).
- `tanda_XX.csv`, `corridas_tXX.csv`, `resultados_tXX.csv` — cada tanda: qué se mandó y qué volvió.
- `mapa_kmc.{npy,png}` — reconstrucción TPS; `dip/final.{npy,png}` — final TPS + DIP.
- `control.{csv,json,png}` — 32 puntos de control no usados y el error contra el mapa final.
- `estado.png`, `avance.png`, `auto.log` — seguimiento del loop.
- `dip_v1/`, `dip_64/`, `dip_pre_*/`, `control_v1.*`, `viejo/` — versiones anteriores, guardadas para comparar.

Los mismos 13 g rehechos con el binario corregido: `autoexp/kmc_ef/` (solo en Mendieta por ahora, desde 19/10).

## g = −4: mapas densos 64×64

- `g-4/grid64/` — crudo 64×64 con el binario viejo (4096 puntos). Figura final: `final_lector/crudo.png`.
- `g-4/grid64_refresco/` — crudo 64×64 con el binario corregido (en curso desde 19/10).
  `adelanto_1910/` = primer adelanto (153 puntos).

## Análisis aparte

| Carpeta | Qué es |
|---|---|
| `investigacion_soc0/` | Los errores del KMC: `NOTAS.md`, `para_profes.txt`, fuentes corregidas (`kmc_ckpt_refresco_ef.cpp` = la versión en uso), corridas de prueba (`run_*`, `comparacion_ef/`). |
| `tendencia/` | Transectas sobre la diferencia KMC − continuo (12/10) y `todos_finales.png` (los 13 g en una figura). |
| `comparacion_random/` | Procedimiento vs. puntos al azar en el continuo (`psnr.png`, `mapas.png`). |
| `pseudo_por_g.csv` | Puntos "sospechosos"/cortados por g (la anomalía de log Ξ alto). |
| `_backup_64_1004_1919/` | `kmc.json` de antes de subir el presupuesto a 96 (4/10). |
| `arranque.log`, `rehacer_final.log` | Logs de relanzamientos. |
