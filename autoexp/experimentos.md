# Bitácora de autoexp

Objetivo: ≤ 64 puntos, > 38 dB. Puntaje = PSNR medio en g ∈ {-4.0, -2.0, -0.5} (peor g entre
paréntesis). Lo más reciente va arriba. La tabla completa está en `leaderboard.csv`.

## Referencias previas (no hechas con este evaluador)

- DIP `grid` + reg0.08, g=-4.0: 48 pts → 20.6 dB; 120 pts → 27.2 dB.
- DIP `frontier_mix` (oráculo) + reg0.08, g=-4.0: 164 pts → 38.7 dB; ~82 pts → 14.4 dB.
- physics-calibration (branch aparte), g=-4.0, 64 pts, 64×64: 34.1 dB. Ojo: la verdad sale
  del mismo modelo, así que es optimista.

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
