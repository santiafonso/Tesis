# Bitácora de autoexp

Objetivo: ≤ 64 puntos, > 38 dB. Puntaje = PSNR medio en g ∈ {-4.0, -2.0, -0.5} (peor g entre
paréntesis). Lo más reciente va arriba. La tabla completa está en `leaderboard.csv`.

## Referencias previas (no hechas con este evaluador)

- DIP `grid` + reg0.08, g=-4.0: 48 pts → 20.6 dB; 120 pts → 27.2 dB.
- DIP `frontier_mix` (oráculo) + reg0.08, g=-4.0: 164 pts → 38.7 dB; ~82 pts → 14.4 dB.
- physics-calibration (branch aparte), g=-4.0, 64 pts, 64×64: 34.1 dB. Ojo: la verdad sale
  del mismo modelo, así que es optimista.

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
