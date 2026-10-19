# Investigación: SoC ≈ 0 arriba del diagrama (log Ξ alto) — 13/10

## El síntoma

En el 64×64 de g=-4 (y en el punto log Ξ 1.43 / log ℓ -3.53 de los 13 g), las corridas de la franja
de arriba terminan con SoC ≈ 1e-4, E ≈ 0.25–0.29 V todo el tiempo y nunca llegan al corte (-0.149 V).
El continuo da ~1 ahí. No son NaN: los NaN/inf que aparecen son las 5 filas basura del final de
**todos** los `datos-*.dat` (también en corridas normales), y el lector las descarta.

## Hipótesis: tasas viejas en la superficie (versión optimizada del código)

En el modelo, el potencial `mui` se resuelve (Brent, `potencial()`) para que la corriente neta
esperada de adsorción − desorción sea la impuesta (`it`), y **todas** las tasas de adsorción/desorción
de la superficie dependen de `mui`:

    ads(s) = KADS_FACTOR · exp((mui − E_ads)/2kT)      des(s) = KADS_FACTOR · exp(−(E_des + mui)/2kT)

En la versión optimizada (`KMC-Galvanostatic_noclus_param_claude.cpp`, la que corre en Serafín y la
misma que mandó el profe), después de cada evento que cambia la superficie se recalcula `mui`, pero
el Fenwick solo se actualiza para el sitio tocado y sus vecinos (`VelocidadesAds` / `VelocidadesDif`).
Los otros ~1600 sitios de la superficie quedan con tasas calculadas con un `mui` viejo. La
retroalimentación galvanostática (el potencial que "empuja" la corriente) deja de actuar sobre casi
toda la superficie.

Cuándo importa: cuando la adsorción/desorción es rápida comparada con la difusión hacia adentro
(`KADS_FACTOR = k0/ds`, con `k0 ∝ Ξ·sqrt(ℓ)`; `KDIF_FACTOR = D/a² ≈ 3.5e4 s⁻¹`). En log Ξ = 2,
log ℓ = -1.33: KADS ≈ 3.8e4 s⁻¹ ≈ KDIF. Con log Ξ = 0 es 100 veces menor. Un ion que se adsorbe ve su
propio sitio con el `mui` nuevo (desorción rápida), pero el resto de la superficie sigue adsorbiendo
con la tasa vieja, baja: se desorbe antes de difundir y el SoC no crece.

## Prueba

`kmc_orig.cpp` = fuente exacta que corrió (commit 745b909). `kmc_refresco.cpp` = lo mismo + 
`RefrescarSuperficie()` después de cada `potencial()` (recalcula ads/des de los 1600 sitios de la
superficie con el `mui` actual). Mismo binario de entrada, `-DFIXEDSEED`, 1 hilo, g=-4.

Resultados: ver abajo.

### Resultados (13/10, PC local, en curso)

**Punto anómalo log Ξ = 2, log ℓ = -1.33** (dt nominal = 1.14 s → la carga lineal da SoC ≈ bin/200):

| bin | SoC esperado | orig | refresco |
|---|---|---|---|
| 2  | 0.010 | 0.0004 (E 0.25) | 0.011 (E 0.10) |
| 12 | 0.060 | 0.0004 | 0.055 (E 0.076) |
| 25 | 0.125 | 0.0004 | 0.141 (E 0.061) |

orig termina en 2 min con SoC final 0.0004 (igual que en Serafín). refresco sigue la corriente impuesta.

**Punto normal log Ξ = 0, log ℓ = -1** (control): las dos versiones dan casi lo mismo (bin 42: SoC 0.212 vs
0.207, mismo E). El efecto solo aparece donde la adsorción/desorción es rápida (Ξ alto).

Conclusión preliminar: el SoC ≈ 0 de arriba viene de las tasas de superficie con `mui` viejo, no es física
ni NaN. **OJO (corregido 13/10 19:25): NO es de la optimización.** La versión vieja sin optimizar
(Mendieta `~/tesis_kmc_ml/monte_carlo/KMC-Galvanostatic-interactions-terminal.cpp`, abril, g=0 fijo; copia en
`version_vieja/`) tiene la misma lógica: `VelocidadesAds` actualiza solo sitio + vecinos y `Velocidades()`
completo solo al inicio. Está en el algoritmo original → preguntar a los profes si es intencional. Probablemente explica también la franja "atrasada" log Ξ 0.9–1.8 (SoC 10–15 % por
debajo de la corriente impuesta). Falta: terminar las corridas, ver el SoC final y el costo de tiempo del
refresco, y decidir con los profes si se corrige el código y se rehace la zona afectada.

### Corridas en Serafín (13/10 19:35, step 432112.4, nice 19, tope 20 h)

`~/kmc/investigacion_soc0/` (`run.sh`, una carpeta `run_<nombre>/` por corrida con `estado.txt`):
`vieja_g0_xi2_el-1.33` (LA CLAVE: ¿la versión vieja también da SoC ≈ 0?), `orig_g0_xi2_el-1.33`,
`refresco_g0_xi2_el-1.33`, `refresco_g-4_xi2_el-1.33`, `orig_g-4_xi0_el-1`, `refresco_g-4_xi0_el-1`.
Ver: `ssh -J siaosorio@mendieta.ccad.unc.edu.ar siaosorio@serafin.ccad.unc.edu.ar` y
`for d in ~/kmc/investigacion_soc0/run_*; do echo $d; cat $d/estado.txt; grep -vi "inf\|nan" $d/datos-* | tail -1; done`
(SoC = 1ª columna; la versión vieja escribe todo al final y su archivo se llama `datos-40x40x40-Chi-...`).

Ya terminado local: `orig_g0_xi2_el-1.33` → SoC final 0.0005 (no depende de g). Las corridas locales de
refresco se cortaron a mano (se pasaron a Serafín).

Mensaje para el profe (mandar SOLO si la vieja también da SoC ≈ 0):
> Encontré de dónde viene el SoC 0, y está también en la versión sin optimizar: cuando el potencial cambia,
> solo se actualizan las tasas de adsorción/desorción del sitio y sus vecinos, no las de toda la superficie.
> Probé actualizarlas todas y en esa zona carga normal; en un punto normal da casi igual. ¿Es intencional o
> habría que actualizar toda la superficie?

### 14/10: resultado de la versión vieja → la causa completa

- La 1ª corrida vieja (step 432112.4) se perdió: al sacarle el `timeout` a mano terminó el step y Slurm la
  mató (17 h, escribe todo al final). Relanzada con una línea de progreso (`version_vieja/kmc_vieja_progreso.cpp`,
  solo `fprintf(stderr)` cada 1e5 pasos), sin tope: Serafín `run_vieja2_g0_xi2_el-1.33/progreso.log`, step 432112.9.
- **La versión vieja SÍ carga** en el punto anómalo (g=0, log Ξ 2, log ℓ -1.33): a t = 0.004 s (1.6 % de dt)
  SoC 0.037, E ≈ 0.07 V (como `refresco`). Nuestra versión: SoC 0.0005 todo el tiempo.
- Por qué, si las dos actualizan solo sitio + vecinos: **el orden de inicialización.** Las tasas de los sitios
  de superficie que nunca se tocan quedan congeladas con el `mui` con que se calcularon al principio.
  - vieja: `Inicial2()` → `Velocidades()` con `mui = 0` (valor inicial de la variable global) → congeladas en E = 0 V
    → se adsorbe bien.
  - optimizada (`kmc_orig.cpp` l. 287-292): `mui = muoff-1; potencial(); Velocidades();` → congeladas con el `mui`
    resuelto con la superficie vacía (E ≈ +0.35 V) → adsorción casi nula en Ξ alto → SoC ≈ 0.
- Conclusión: el problema de fondo (tasas viejas) está en el algoritmo original; el síntoma SoC ≈ 0 lo dispara
  el orden de inicialización de la versión optimizada. En los dos casos el resultado depende de un `mui`
  inicial arbitrario. `RefrescarSuperficie()` (tasas de toda la superficie con el `mui` actual) no depende de eso.

Resultados Serafín (13/10, terminados): punto anómalo g=-4 refresco SoC final 0.957 (orig 0.0004); g=0 refresco
0.945 (orig 0.0005); punto normal (log Ξ 0, log ℓ -1) orig 0.965 vs refresco 0.962. Costo: refresco 2-20× más
lento (anómalo 7062 s vs 338 s; normal 3034 s vs 1660 s).

### 14/10 13:13: tanda de comparación contra el 64×64 (versión corregida, con checkpoint)

40 puntos de la grilla 64×64 de g=-4: filas log Ξ = 2, 1.71, 1.43, 1.14, 0.86, 0.57, 0.29, 0 (r 0,3,…,21) ×
columnas log ℓ = -2, -1.52, -1.05, -0.48, 0 (c 21,26,31,37,42), 1 réplica. `comparacion_grid/tanda_{a,b}.csv`.
Binario: Serafín `~/kmc/refresco/kmc_ckpt` = checkpoint + `RefrescarSuperficie()` (fuente `kmc_ckpt_refresco.cpp`;
el `~/kmc/kmc_ckpt` de la grilla y los loops NO se tocó). Corren con `kmc_largo.sh` (KMC_DIR=~/kmc/refresco)
en `~/kmc/tandas/g-4_refresco_{a,b}/` (20 núcleos cada una, contenedores 432113 / 432114), relevo desde el login
de Mendieta con `scripts/kmc_largo_relevo_refresco.sh` (copia del relevo con KMC_DIR), logs
`autoexp/kmc/investigacion_soc0/relevo_{a,b}.log` en Mendieta. **Cuando haya contenedores nuevos, relanzar:**
`NREP=1 setsid nohup scripts/kmc_largo_relevo_refresco.sh "~/kmc/tandas/g-4_refresco_a" 20 "<id>" >> autoexp/kmc/investigacion_soc0/relevo_a.log 2>&1 < /dev/null &` (ídem b).
Comparar con `grid64/final_lector/mapa_corregido.npy` en los mismos (row, col).
