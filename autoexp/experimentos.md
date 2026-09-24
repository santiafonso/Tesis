# Bitácora de autoexp

Objetivo: ≤ 64 puntos, > 38 dB. Puntaje = PSNR medio en g ∈ {-4.0, -2.0, -0.5} (peor g entre
paréntesis). Lo más reciente va arriba. La tabla completa está en `leaderboard.csv`.

## Referencias previas (no hechas con este evaluador)

- DIP `grid` + reg0.08, g=-4.0: 48 pts → 20.6 dB; 120 pts → 27.2 dB.
- DIP `frontier_mix` (oráculo) + reg0.08, g=-4.0: 164 pts → 38.7 dB; ~82 pts → 14.4 dB.
- physics-calibration (branch aparte), g=-4.0, 64 pts, 64×64: 34.1 dB. Ojo: la verdad sale
  del mismo modelo, así que es optimista.

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
