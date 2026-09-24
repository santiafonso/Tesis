# Bitácora de autoexp

Objetivo: ≤ 64 puntos, > 38 dB. Puntaje = PSNR medio en g ∈ {-4.0, -2.0, -0.5} (peor g entre
paréntesis). Lo más reciente va arriba. La tabla completa está en `leaderboard.csv`.

## Referencias previas (no hechas con este evaluador)

- DIP `grid` + reg0.08, g=-4.0: 48 pts → 20.6 dB; 120 pts → 27.2 dB.
- DIP `frontier_mix` (oráculo) + reg0.08, g=-4.0: 164 pts → 38.7 dB; ~82 pts → 14.4 dB.
- physics-calibration (branch aparte), g=-4.0, 64 pts, 64×64: 34.1 dB. Ojo: la verdad sale
  del mismo modelo, así que es optimista.

## 2026-09-24 (16) — "no sabemos la forma de antemano en KMC": prueba fuera de muestra ★

Todo se venía afinando con los 9 g negativos del modelo del continuo (acantilado a 0, borde
recto, monotonía, zona que se desvanece vertical). Prueba en mapas NUNCA usados: diagramas g=+2
y g=+4, e imágenes de forma distinta (Gauss, Himmelblau, Rosenbrock). El oráculo ahora acepta
imágenes sintéticas por nombre.

| mapa | grilla + TPS | receta vieja | **receta que chequea supuestos** |
|---|---|---|---|
| g=+2 | 32.35 | 35.09 | **36.62** |
| g=+4 | 34.53 | **39.33** | 38.37 |
| Gauss | **50.21** | 43.50 | 49.28 |
| Himmelblau | 28.12 | 23.09 | **30.60** |
| Rosenbrock | 23.71 | 22.99 | **24.93** |
| media | 33.78 | 32.80 | **35.96** |

La receta vieja **empeoraba** en formas distintas (Gauss −6.7 dB: la proyección monótona
deformaba una campana; la bisección gastaba consultas en los bordes a 0 de la imagen).
Cambios (`interp.auto`, `sampling.bisect(check_shape=True)`):
- **Muestreo:** si los 30 puntos de la grilla ya violan la monotonía (tol 0.02 + 3σ), el
  modelo de acantilado no aplica. No se hace bisección y todo el resto va al relleno LOO
  **genérico** (TPS común en todo el mapa). En los 9 g negativos el muestreo sale idéntico
  (verificado).
- **Reconstrucción:** monotonía y cotas solo si ningún par de puntos la viola. Acantilado solo
  si ≥ 3 puntos quedan debajo de la recta y ≥ 90 % valen ~0. Compresión vertical 0.6 o 1.0
  elegida por LOO sobre los puntos. Si no hay acantilado: TPS común (+ monotonía si aplica).
- En g=+2/+4 decidió "sin acantilado, con monotonía": los diagramas repulsivos no llegan a 0
  exacto con salto.

## 2026-09-24 (15) — planificador de tandas para KMC, σ=0.05, perfil de rampa (no)

- **`autoexp/kmc_planner.py`**: init / next / add / reconstruct / simulate. Oráculo de
  "repetición": en cada `next` se vuelve a correr el muestreo con los resultados ya cargados y
  se corta en los puntos que faltan (la tanda siguiente). Bisección pasada a rondas en lote
  (mismos puntos que de a uno, verificado). Simulación: **9 tandas** [30, 5, 5, 5, 5, 4-5, 4,
  4, 1-2]; PSNR g=-4 46.2, g=-2 38.6, g=-0.5 39.4; g=-2 con σ=0.01: 37.6.
- Ojo: al volver el suavizado 1e-4 el default, el LOO interno también lo usaba y rendía menos
  (9 g: 39.98 (35.69), g=-3.5 38.4). El muestreo usa ahora la TPS sin suavizar (sin ruido) y
  reproduce exactamente VAL9_ransac_s1e-4: 40.62 (36.20).
- **σ=0.05 con DIP:** TPS robusta 31.41 (30.74), fusión + guarda (0.12) **32.62 (31.24)**, SSIM
  0.939 → 0.968. **Lo que suma DIP crece con el ruido:** +0.7 (σ .01), +0.7 (.03), +1.2 (.05) dB.
- **Perfil de rampa juntando todas las columnas** (regresión isotónica en d + TPS del residuo):
  33-34 contra 40.2. Mezcla la izquierda (meseta ~1) con la derecha (zona que se desvanece
  ~0.2). Descartado.

## 2026-09-24 (14) — DIP con ruido, robustez v2, sondas de rampa (otra vez no)

**DIP con ruido** (cluster, híbrido con parámetros robustos; 4 g). Fusión con guarda
0.02 + 2σ (el residuo en los puntos reales incluye el propio ruido):

| σ | TPS robusta sola | DIP solo | **fusión + guarda** | SSIM TPS → fusión |
|---|---|---|---|---|
| 0.01 | 38.31 (35.02) | 35.77 (33.76) | **39.02 (35.48)** | 0.983 → 0.990 |
| 0.03 | 34.23 (32.44) | 32.96 (31.28) | **34.89 (33.64)** | 0.963 → 0.982 |

→ Con ruido, DIP solo pierde contra la TPS robusta, pero **la fusión suma ~0.7 dB de media, más
en el peor g (+1.2 con σ=0.03), y sobre todo en SSIM**: DIP filtra el ruido que la TPS copia.

**Robustez a la grilla (receta actual sin DIP, 9 g):** offset 0.35: 38.41 (35.84); 0.50:
40.62 (36.20); 0.65: 37.68 (35.63). **Número honesto: ~38.9 ± 1.3 dB de media; peor g entre
35.6 y 36.2**, que casi no depende de la grilla (antes ~37.5).

**Sondas de rampa condicionadas** (solo si el frente es suave, d=5/12, 6 o 4/10 en 2-3
columnas): 39.2 / 39.5 / 38.4 contra 40.2 sin sondas. Segunda vez que no sirven: los puntos
valen más en el relleno LOO. Revertido.

## 2026-09-24 (13) — TPS suavizada + fusión: 41.1 dB en 9 g, 8 de 9 ≥ 38 ★

- El suavizado 1e-4 de la TPS mejora incluso sin ruido (se vio al comparar sigma_0 = 39.5
  contra ransac_noise0 = 40.2 en 4 g). En 9 g, TPS sola: **40.62 (36.20), 8 de 9 g ≥ 38**
  (antes 40.31 (35.09)). Queda como default de `interp.cliff`.
- Fusión con guarda + TPS suavizada, 9 g: **B=20: 41.07 (37.25), 8 de 9 g ≥ 38**; B=30:
  41.20 (37.08). Por g (B=20): -4 44.8, -3.5 44.4, -3 43.7, -2.5 38.6, -2 38.3 (solo TPS,
  guarda), **-1.5 37.2**, -1 39.9, -0.5 41.1, 0 41.6. Solo g=-1.5 queda debajo de 38.
- **Parámetro único `sigma`** (ruido de KMC estimado) en `bisect` y `cliff` que fija los
  umbrales: eps = max(1/255, 3σ), salto mínimo del muestreo = max(0.1, 6σ), steep = max(0.1, 5σ).
  Con σ=0 no cambia nada. En el muestreo, un salto mínimo de 0.1 rinde más que 0.05 aun con
  σ=0.01 (38.3 contra 36.7).

## 2026-09-24 (12) — fusión validada en 9 g + guarda de falla de DIP + Optuna profes

**TPS + RANSAC en 9 g** (sin ruido): 40.31 (35.09); antes 40.17 (34.86).

**Fusión en 9 g** (DIP híbrido del cluster + TPS actual, B=20):

| | -4 | -3.5 | -3 | -2.5 | -2 | -1.5 | -1 | -0.5 | 0 | media (peor) |
|---|---|---|---|---|---|---|---|---|---|---|
| TPS + RANSAC | 45.9 | 43.7 | 42.3 | 40.0 | 36.5 | 35.1 | 39.0 | 40.6 | 39.9 | 40.31 (35.09) |
| fusión DIP tesis | 44.9 | 44.3 | 43.5 | 38.9 | 32.5 | 36.0 | 40.2 | 41.3 | 41.6 | 40.34 (32.46) |
| fusión DIP intermedio | 26.6 | 43.2 | 40.4 | 35.1 | 37.0 | 35.6 | 39.4 | 42.1 | 40.0 | 37.71 (26.59) |

→ La fusión gana en 6 de 9 g, pero cuando DIP falla arrastra todo. **Detector de falla sin
mirar el mapa:** RMS del residuo de DIP en los puntos REALES consultados de arriba del
acantilado. Cuando anda es ≤ 0.008; cuando falla, ≥ 0.03 (tesis g=-2: 0.034; intermedio
g=-4: 0.156 y g=-2.5: 0.032). **Guarda 0.02 → solo TPS en ese g:**

| | 9 g: media (peor) |
|---|---|
| **fusión DIP tesis + guarda** | **40.79 (36.01)**: mejor que la TPS en las dos métricas |
| fusión DIP intermedio + guarda | 40.40 (35.60) |

(El umbral se eligió mirando estos datos, pero el hueco entre 0.008 y 0.03 es grande.)

**Estudio Optuna `profes_grid_uniform` (181 trials, pedido 1-3 del 17/9):** mejor DIP con
grilla de 64 puntos ≈ **29.9 (26.2)**. Todos los mejores: meshgrid, skip 0, width 128, 5
escalas, LR 0.005-0.017, ~3000 it, reg 0.015-0.07 (= la config "vase" del paper, afinada).
Uniform nunca aparece arriba. Con los mismos puntos, la TPS da 24.4: DIP le gana por +5.5 dB,
pero el muestreo nuevo + TPS da 40+.

**`autoexp_v1` roto:** cambié las opciones de una categórica (`n1`) con el estudio empezado y
Optuna rechaza el trial al instante: 49 192 trials fallidos en bucle (GPU quemada ~1.5 h,
cancelado). Arreglado en `autoexp_v2`: muestreo fijo, objetivo = fusión con guarda, y el
worker se corta tras 10 fallos seguidos. **Regla: no cambiar el espacio de un estudio
existente; estudio nuevo.**

## 2026-09-24 (11) — ruido tipo KMC + RANSAC

Oráculo con ruido opcional (`noise`: gaussiano fijo por pixel, recortado a [0,1]); el puntaje
sigue siendo contra el mapa limpio. 4 g (-4, -2, -1.5, -0.5):

| config | σ=0 | σ=0.01 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|
| grilla 8×8 + TPS | — | 24.3 (21.1) | 23.8 (20.7) | — |
| receta sin adaptar | 39.5 (35.1) | 30.2 (24.5) | 26.4 (23.3) | — |
| umbral de cero = 3σ + TPS suavizado 1e-4 | — | 36.7 (35.0) | 25.9 (23.9) | — |
| **+ recta por RANSAC** | **40.2 (36.2)** | 36.7 | 27.6 (23.9) | — |
| **+ salto mínimo ~5σ + RANSAC sin bordes duplicados** | — | **38.3 (35.0)** | **34.2 (32.4)** | **31.4 (30.7)** |

- RANSAC también mejora sin ruido (g=-2: 36.5 → 38.3).
- Con ruido, un solo punto ruidoso en la cola de la zona que se desvanece (0.12 donde el real
  es ~0.02) disparaba una bisección falsa que gastaba 11 consultas. Por eso los umbrales de
  "salto real" tienen que escalar con σ, y σ hay que estimarlo en KMC (NRUNS).
- **Pendiente:** DIP con ruido (se esperaría que degrade menos que la TPS) y la fusión.

**DIP híbrido en 9 g** (receta de la tesis, sin fusionar): 37.0 (31.3). Falta fusionarlo con la
TPS (`autoexp.fuse`, B=20). La config intermedia (VAL9_hib_mid) todavía estaba corriendo.

## 2026-09-24 (10) — cotas por monotonía, criterio de relleno, re-barrido de grilla

- **Cotas exactas por monotonía** (`interp.monotone_bounds`): un punto consultado abajo a la
  derecha de p acota f(p) por abajo, y uno arriba a la izquierda por arriba (0 violaciones
  en el mapa real). Recortar la TPS a [L, U] antes de proyectar: +0.2 dB en cada g
  (4 g: 39.33 (34.86) → 39.50 (35.09)). Queda activado.
- **Criterio de relleno por ancho de cotas** (U−L, incertidumbre exacta) contra LOO:
  LOO 39.5 (35.1), cotas 38.7 (35.5), producto 38.5 (35.1). Queda LOO. Ojo: si el LOO usa
  internamente la reconstrucción con mono/cotas, elige peores puntos (37.4); el muestreo
  quedó aislado con su configuración validada.
- **Re-barrido de grilla** con la reconstrucción nueva (4 g):

| n1 / nx | media (peor) | g=-4 |
|---|---|---|
| 24 / 6 | 38.0 (35.0) | 41.7 |
| **30 / 6** | **39.5 (35.1)** | 45.9 |
| **36 / 6** | 39.5 (**36.6**) | 46.1 |
| 42 / 6 | 35.6 (34.0) | 38.5 |
| 30 / 5 | 32.3 (30.4) | 30.4 |
| 35 / 7 | 34.6 (32.5) | 32.5 |
| 40 / 8 | 32.6 (28.8) | 28.8 |

  Con 7-8 columnas **no falla el detector** (la recta sale igual, pendiente −0.49 a −0.51):
  se acaba el presupuesto. 35-40 de grilla + ~5 consultas de bisección por columna > 64, no
  queda relleno y algunas bisecciones no terminan. Sin relleno nadie mide la transición
  vertical de x≈85. **Regla para KMC: elegir las columnas según el presupuesto, dejando ~10
  puntos para el relleno.** Con 5 columnas, al revés: el frente queda mal muestreado.

## 2026-09-24 (9) — franja adaptativa + fusión con la TPS nueva

**Franja de la rampa adaptativa:** v_edge = mediana de los valores consultados a ≤2.5 px
arriba del acantilado (g=-4: 0.81, g=-2: 0.38, g=-1.5 y -0.5: ~0.10). Frente abrupto → along
0.5, band 15; suave → along 0.3, band 25 (interpolado lineal, v_sharp 0.7). Con franja fija
había un compromiso: más ancha ayudaba a los g suaves (-1.5: 33.7 → 35.6) y rompía g=-4
(45.7 → 42.1).

| corrida | 4 g (-4,-2,-1.5,-0.5): media (peor) | 9 g: media (peor) |
|---|---|---|
| franja fija 15 / 0.5 | 38.7 (33.7) | 39.3 (33.7) |
| **adaptativa** | **39.1 (35.3)** | **40.2 (34.9)**: 7 de 9 g ≥ 38 (-4 45.7, -3.5 43.5, -3 42.2, -2.5 40.0, -1 38.8, -0.5 40.4, 0 39.8; -2 36.3, -1.5 34.9) |

**Fusión (TPS adaptativa + DIP del híbrido), dev:**

| DIP | B=6 | B=10 | B=20 |
|---|---|---|---|
| receta tesis | 41.1 (38.7) | 41.5 (39.5) | **41.8 (39.5)** |
| intermedia (LR .003, reg .05, skip 0, nearest, 6000 it) | 41.4 (39.7) | **41.7 (39.8)** | 41.7 (39.6) |
| "vase" | 40.9 (37.5) | 41.0 (37.6) | 41.0 (37.7) |

→ Los 3 g de desarrollo quedan ≥ 39.5 dB con 64 puntos. En validación de 9 g: DIP tesis
(1178757) e intermedia (val9_hib_mid) → `autoexp.fuse`.

## 2026-09-24 (8) — monotonía + híbrido DIP/TPS ★

**Monotonía:** el mapa real es exactamente no creciente hacia abajo y hacia la derecha en
todos los g (0 violaciones: SoC_max baja al crecer ℓ y Ξ). Proyección de la reconstrucción
sobre ese conjunto (Dykstra, alternando regresión isotónica por columnas y por filas):

| corrida | dev: media (peor) | 9 g: media (peor) |
|---|---|---|
| TPS (LOO n30) | 39.6 (36.1) | 38.65 (33.2) |
| **TPS + monotonía** | **40.4 (36.8)** | **39.3 (33.7)**: 5 de 9 g ≥ 38 |

(El muestreo LOO usa internamente la reconstrucción sin proyectar: es cara y no aportaba.)

**Híbrido** (job 1178753; muestreo LOO n30 + pseudo-ceros debajo del borde + franja TPS de 6 px
como pseudo-puntos + DIP):

| DIP | media (peor) | -4 / -2 / -0.5 |
|---|---|---|
| receta de la tesis (LR 1e-3, reg .08, skip 4) | 39.3 (38.0) | 41.7 / **38.0** / 38.1 |
| "vase" del paper | 35.4 (30.8) | 30.8 / 36.5 / **38.9** |
| + monotonía post-hoc | +0.07: DIP ya sale liso | |

**Fusión por distancia al acantilado** (`autoexp/fuse.py`, en local sobre las salidas de DIP):
final = mono(w·TPS + (1−w)·DIP), w = exp(−(d/B)²):

| B (px) | DIP tesis: media (peor) | DIP vase |
|---|---|---|
| solo DIP | 39.3 (38.0) | 35.5 (30.8) |
| 3 / 6 | 40.6 (38.2) / 40.7 (38.3) | 40.5 (37.1) / 40.6 (37.0) |
| **10** | **40.9 (38.6)**: 45.1 / 38.6 / 38.9 → **los 3 g ≥ 38** | 40.6 (36.8) |
| 20 / 40 | 41.2 (38.0) / 41.0 (37.4) | 40.7 (36.6) / 41.0 (36.9) |
| solo TPS | 40.4 (36.8) | |

→ **Primera configuración con todos los g de desarrollo por encima de 38 dB con 64 puntos: TPS
pegada al borde + DIP lejos de él.** Cada uno solo no llega. En validación (job 1178757):
DIP híbrido en los 9 g → fusión local.

## 2026-09-24 (7) — DIP con 64 puntos (Mendieta) y proceso gaussiano (local)

**DIP con los mismos 64 puntos de la grilla 8×8** (dev g; en la columna, la interpolación
TPS con esos puntos da 24.4 (21.2)):

| config DIP | media (peor) | por g |
|---|---|---|
| receta de la tesis: LR 0.001, reg 0.08, skip 4, bilinear, 8000 it | 25.0 (21.9) | -4 21.9, -2 23.9, -0.5 29.3 |
| paper "kate": noise 32, LR 0.01, reg 0.03, skip 128, nearest, 6000 it | 18.1 (13.7) | -4 13.7, -2 17.4, -0.5 23.1 |
| **paper "vase": meshgrid, LR 0.01, reg 0.03, skip 0, nearest, 5000 it** | **28.5 (25.3)** | -4 25.3, -2 26.7, -0.5 33.3 |
| paper "vase" + uniform (seed 0) | 22.2 (19.4) | |

→ **Con los hiperparámetros del paper para agujeros grandes, DIP le gana por +4 dB a la
interpolación con los mismos puntos. La receta de la tesis no.** La receta venía afinada para
~328 puntos; con 64, lo que manda es una red sin skips (más suave) y un LR 10 veces más alto.

**DIP con el muestreo del acantilado** (`autoexp_v1`, relleno por gradiente viejo): 27.8 a 28.7,
fallando en g=-4 (~18.5: DIP deja un hueco arriba del borde, sin puntos). Pendiente: el híbrido
(job 1178753): muestreo LOO + ceros + franja clásica de 6 px junto al borde + DIP
{tesis, vase, intermedia}, solo o promediado 50/50 con la clásica.

**Proceso gaussiano** (Matérn anisótropo, escalas aprendidas) en lugar de la TPS arriba del
acantilado: **peor**. nu=2.5: 31.7 (22.4); nu=1.5: 35.8 (30.6); contra TPS 39.6 (36.1). Con tan
pocos puntos, las escalas aprendidas quedan mal y deja agujeros entre los datos. Revertido.

## 2026-09-23 (6) — relleno por validación cruzada (leave-one-out) ★

En vez de rellenar donde el gradiente es alto: cada punto de arriba del acantilado se predice
con los demás (reconstrucción `cliff` completa), el |error LOO| se interpola a toda la imagen
y los puntos nuevos van, en tandas de 4, donde ese error es alto y lejos de lo consultado. Va
directo a donde el modelo se equivoca, sin suponer que el error vive donde hay gradiente.

| corrida | dev: media (peor) | 9 g: media (peor) |
|---|---|---|
| cliffL_n36_b4 | 38.6 (35.0) | — |
| cliffL_n36_b8 | 37.0 (35.0) | — |
| **cliffL_n30_b4** | **39.6 (36.1)**: g=-4 44.9, -2 36.1, -0.5 38.0 | — |
| **VAL9_loo_n30 offset 0.5** | — | **38.65 (33.2)**: ≥38 en g=-4 (44.9), -3.5 (42.8), -3 (41.4), -2.5 (38.0); -0.5 37.95 |
| VAL9_loo_n30 offset 0.35 | — | 37.3 (32.7) |
| VAL9_loo_n30 offset 0.65 | — | 36.5 (33.6) |

**Promediando offsets, ~37.5 dB en los 9 g con 64 puntos** (antes ~34.7). El objetivo de 38 dB
de media se cumple con la grilla centrada. Todavía no en todos los g: el peor sigue siendo
g=-1.5 (~33). Grilla más chica (30 en vez de 36) + más relleno LOO rinde más: los puntos
elegidos por error valen más que los de la grilla fija.

## 2026-09-23 (5) — robustez: ¿la receta depende de dónde cae la grilla?

Receta actual (grilla 6×6 + bisección del acantilado + relleno `cliff` gap 4 + `cliff` con
rampa alineada y vert 0.6), validada en los 9 g corriendo la grilla entera:

| offset de la grilla | 9 g: media (peor) |
|---|---|
| 0.20 | 32.6 (30.3) |
| 0.35 | 34.2 (29.9) |
| **0.50** (con el que se afinó todo) | **36.2 (31.9)** |
| 0.65 | 35.8 (31.2) |
| 0.80 | 34.9 (29.6) |

**Lectura honesta:** el método da **~34.7 ± 1.3 dB de media en los 9 g con 64 puntos**
(promedio sobre offsets). El 36.2 está algo favorecido porque todo se afinó con offset 0.5. Lo
robusto es la mejora contra los baselines, que se mantiene con cualquier offset: TPS en grilla
24.4, DIP con 48 a 82 puntos 14 a 35 según el g. Para la tesis conviene reportar la banda, no
el mejor caso.

## 2026-09-23 (4) — ajuste fino del modelo de acantilado + pedidos de la reunión

| corrida | qué | dev: media (peor) | 9 g: media (peor) |
|---|---|---|---|
| cliffK_* | núcleo base: tps / cubic-rbf / linear-rbf / griddata cúbica, suavizado 1e-5 a 1e-4 | TPS sin suavizar sigue siendo el mejor (35.5) | — |
| cliffV_0.6 | TPS arriba del borde con la distancia vertical × 0.6 (la zona que se desvanece es una transición que depende casi solo de x) | **36.6 (33.9)** | **36.2 (31.9)**: g=-4 **41.7**, -3.5 **41.1**, -3 **38.4** |
| cliffV_0.5 / 0.4 / 0.25 / 0.15 | más compresión | 36.5 / 36.5 / 35.0 / 31.4 | 0.5: 36.1 (31.7) |
| cliffP_* | sondas del perfil de la rampa a d px sobre el borde, en 2-3 columnas | 33.4-35.9: **peor**, le quitan presupuesto al relleno. Revertido | — |

Error restante (VAL9_cliffF_gap4): ~88 % en la rampa y la zona intermedia, ~12 % en la meseta,
0 % debajo del borde.

**Pedidos del 17/9:**
- (1-3) Optuna + grid/uniform con hiperparámetros + grilla como hiperparámetro: estudio
  `profes_grid_uniform` (sampler grid con nx/offset como hiperparámetros, o uniform; solo DIP).
  Semillas: las configuraciones de inpainting del **paper** (notebooks/inpainting.ipynb:
  "vase" = meshgrid, LR 0.01, 5000 it, reg 0.03, skip 0, nearest; "kate" = noise 32, LR 0.01,
  6000 it, reg 0.03, skip 128, nearest) y la receta de la tesis (LR 0.001, reg 0.08, skip 4,
  bilinear). **LR, skip y upsampling del paper nunca se habían probado** (la tesis usa un LR
  10 veces menor). En paralelo, `autoexp_v1` (bisección del acantilado + DIP con pseudo-ceros).
  Job 1178748 (multi, 4 workers, en cola) + 1178749 (short, 2 workers, corriendo desde ~23:40).
- (4) Ventana: la bisección ya concentra ~25 de los 64 puntos en una franja angosta alrededor
  del frente. Falta la variante con DIP solo dentro de la ventana.
- (5) Rosenbrock → 3D: `analysis/rosenbrock_3d.py` → `results/rosenbrock_3d/rosenbrock_3d.png`
  (DIP 1/5/10 % y TPS con 64 puntos, como superficie). Con 1 % DIP recupera la forma pero la
  cresta angosta queda ondulada; recién con 5 % queda fiel.
- (6) scipy vs DIP: tabla de arranque (TPS 24.4 con grilla 8×8).

## 2026-09-23 (3) — modelo de acantilado a cero (local, sin DIP)

Hallazgo al mirar las columnas bisecadas: en **todos** los g el perfil vertical es meseta ≈1 →
rampa suave (corta en g=-4, larga en g=-0.5) → **salto a exactamente 0**. La bisección al nivel
medio caía dentro de la rampa, así que en g=-2/-0.5 el `front_split` nunca encontraba el frente
y caía a TPS común.

| corrida | qué | dev: media (peor) | 9 g: media (peor) |
|---|---|---|---|
| cliff_b36_d1 | bisección del borde v=0 + `cliff` (0 debajo, TPS arriba), eps=0.02 | 28.7 (19.7) | — |
| cliff2_b36_d1 | eps=1/255 (0.02 era un escalón de cuantización del PNG) + salto ≥0.05 + ajuste robusto | 32.8 (30.4) | — |
| cliff3_b36_cm0.03 | salto ≥0.03 (en g=-0.5 el último pixel antes del 0 vale 0.047) | 33.6 (32.7) | 33.2 (28.5): g=-1.5 sin acantilado |
| VAL9_cliffsteep | criterio por pendiente local (algún punto a ≤5 px del lado no nulo vale ≥0.1) | — | 33.4 (29.9) |
| VAL9_cliffA | + rampa interpolada en coords alineadas al borde (along .5, band 15) | 34.5 (33.4) | **34.4 (33.0)**: los 9 g entre 33 y 35 |
| cliffF_n36_cliff | relleno con \|∇\| de la reconstrucción `cliff`, solo arriba del borde | 35.6 (32.7): g=-4 **39.8** | — |
| **VAL9_cliffF_gap4** | ídem, excluyendo 4 px sobre el borde | 35.5 (33.4) | **35.2 (32.2)**: g=-4 **39.1**, g=-3.5 **39.05** |
| VAL9_cliffF_gap2 | gap 2 px | — | 35.0 (32.7) |

Descartados: grillas nx=8/10 o n1=48/50 (28-30 dB; menos columnas bisecadas o menos relleno),
tol=2 en la bisección (peor), relleno mixto (entre medio de los dos).

**Conclusiones:**
- Error restante de VAL9_cliffA: ≥99 % arriba del borde; el borde está bien ubicado (<1 % del
  error en píxeles mal clasificados). Fuentes: la rampa entre columnas bisecadas (se arregló
  con las coords alineadas) y la transición vertical de la zona que se desvanece, en x≈85, que
  cae entre dos columnas de la grilla (se arregló con el relleno `cliff`).
- **Con 64 puntos y sin DIP, los dos g más duros ya pasan los 38 dB** (39.1 / 39.05). Los de
  rampa suave quedan en 32-34. Sensibilidad: el resultado depende de dónde caen las columnas
  de la grilla (nx=6 anda y nx=8 no), así que ojo con el sobreajuste a estos mapas. Por eso
  se valida siempre en los 9 g.

**Siguiente:** Optuna en Mendieta (en cola desde las 21:30) sobre este muestreo: ¿DIP con
pseudo-ceros debajo del borde mejora la parte suave?

## 2026-09-23 (2) — frente explícito + bisección (local, sin DIP)

| corrida | qué | media (peor) |
|---|---|---|
| split_adapt{32,40,48}_deg2 | adaptativo + `front_split` (frente = parábola por los puntos medios de los pares con salto; TPS por lado) | 27.1 / 26.2 / 27.2 — inestable, deja una costura vertical al final del tramo |
| split1_adapt{32,40,48} | ídem, con una recta en todo el ancho | 26.4 / 28.1 / 27.5 (24.3) — la recta queda corrida 1-3 px |
| bisect{25,36,42}_rbf_tps | grilla + **bisección vertical del frente** + resto adaptativo, TPS común | 24.9 / 26.5 / 26.5 |
| **bisect36_front_split** | grilla 36 + bisección + `front_split` (max_len=4) | **30.25 (26.83)** — g=-4: **34.1**, g=-2: 26.8, g=-0.5: 29.9 |
| bisect36_fs_al* | + coordenadas alineadas al frente (along 0.5/0.25/0.1, band 8) | 30.3 (26.8) con split; sin split, peor. **No ayuda**: queda apagado (along=1) |

**Conclusiones:** con presupuesto fijo, gastar ~25 consultas en ubicar el frente a 1 px por
bisección vale más que repartirlas. En g=-4 (el caso "imposible" para DIP con pocos puntos)
esto da 34 dB con 64 puntos. Lo que limita ahora son los frentes suaves (g=-2, g=-0.5): partir
por lados crea un escalón donde en realidad hay una rampa, y entre las columnas sondeadas la TPS
arma "cuentas".

**Siguiente:** usar `bisect36_front_split` como fuente de pseudo-puntos para DIP (que DIP
suavice lo que la TPS deja con cuentas) y sumar sampler=bisect al espacio de Optuna.

## 2026-09-23 — arranque, sin DIP (local)

| corrida | qué | media (peor) |
|---|---|---|
| base_grid8x8_nearest | grilla 8×8 + vecino más cercano | 19.1 (17.7) |
| base_grid8x8_linear | grilla 8×8 + `griddata` lineal | 21.9 (19.6) |
| base_grid8x8_cubic | grilla 8×8 + `griddata` cúbica | 22.9 (20.3) |
| base_grid8x8_rbf_tps | grilla 8×8 + RBF thin-plate | 24.4 (21.2) |
| halton_tps | Halton + TPS | 23.1 (19.9) |
| adapt_n1{16,32,48}_p1_tps | adaptativo: n1 en grilla, resto en tandas de 8 donde \|∇\| × distancia es máximo | 25.1 / 26.8 / **27.8 (23.3)** |
| adapt_*_p2 | ídem con \|∇\|² | peor que p1 |

**Conclusiones:** interpolación de scipy (punto 6 de la reunión): TPS es la mejor, 24.4 dB con
grilla. El muestreo adaptativo en dos etapas suma +3.4 dB sin mirar la imagen: los puntos se
ubican solos a ambos lados del frente. Conviene gastar la mayor parte en la grilla (n1=48). El
error que queda: la TPS ondula sobre el frente y deja manchas en las mesetas.

**Siguiente:** estudio Optuna en Mendieta (`autoexp_v1`): muestreo (grilla nx/offset como
hiperparámetro o adaptativo) × pseudo-puntos de TPS × hiperparámetros de DIP.
