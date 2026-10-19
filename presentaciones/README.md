# presentaciones/

Una carpeta por presentación, nombrada `presentacionN_DD_MM/` (N = número de
presentación, DD_MM = día y mes en que se armó).

Cada carpeta contiene:

- `presentacionN_DD_MM.pptx` — la presentación, editable.
- las figuras resumen generadas para esa presentación (`sweep_summary.png`,
  `iter_curves.png`, `gfix_delta.png`, …).
- `make_presentacionN_DD_MM.py` — copia exacta del script que la generó
  (para poder reproducirla tal cual más adelante).

La 1 usa el generador vivo de la raíz (`PRES_NAME=… ./venv/bin/python
make_presentacion.py`). De la 2 en adelante cada carpeta trae su propio
generador autocontenido (lee las métricas de `archive/` y arma el `.pptx`):

```bash
./venv/bin/python presentaciones/presentacion2_09_09/make_presentacion2_09_09.py
```

## Índice

| Carpeta | Fecha | Contenido |
|---|---|---|
| `presentacion1_02_09/` | 02/09/2026 | DIP sobre el diagrama de fases del modelo del continuo: parámetros de `galpynostatic`, definición de MAE/PSNR/SSIM + el código que las calcula, las 7 corridas de reconstrucción 128×128 (50–99 % de puntos ocultos), tabla resumen y curvas de fidelidad. |
| `presentacion2_09_09/` | 09/09/2026 | Segunda entrega, sobre las cinco cosas que quedaron de la reunión anterior: (1) puntos sobre la frontera vs. repartidos — concentrarlos en la frontera es lo peor; (2) el código ordenado en un paquete `dip/`; (3) barrido fino del % oculto y un valor estándar propuesto (mf 0.98); (4) error vs. iteración — sobreajuste temprano en los casos difíciles; (5) barrido en `g` (Frumkin), los negativos como caso duro, y el intento de arreglo cargando la máscara en la frontera. Incluye: cómo se define la frontera (\|∇SoC\| suavizado) y las 4 familias de muestreo; 6 diapositivas de detalle de frontera (una por régimen de N) con la distribución de puntos y su reconstrucción; y 3 de reconstrucciones del barrido en `g` (imágenes por g, con la frontera marcada). Mismo estilo y estructura que la 1. 29 diapositivas. |
| `presentacion5_05_10/` | 05/10/2026 | Presentación corta para la reunión, con el estilo de las 1-4. Orden: (1) Optuna sobre DIP (181 pruebas, ~30 dB máx. con 64 puntos), (2) interpolación de scipy (sugerencia de Andrés: TPS), (3) estructura física (idea de Maxi): bisección, la línea, cero abajo y monotonía, (4) DIP para suavizar (fusión por zonas); después la receta en una filmina, continuo con 64 y con 96 puntos, KMC real g = −4 y g = +3 con su control, cómo se corre el KMC y preguntas; apéndice con el problema y los 8 pasos con KMC (con fundido). 25 diapositivas. `./venv/bin/python presentaciones/presentacion5_05_10/make_presentacion5_05_10.py` |
| `presentacion6_05_10/` | 05/10/2026 | Para la reunión siguiente a la del 5/10, mismo estilo que la 5: los pedidos de esa reunión y su estado; avance del KMC en los 13 g (−8 a 4); mapas de KMC de los g negativos (−1 a −4, y −5 a −8 en su primera tanda) y positivos, con el continuo solo de referencia; la grilla 64×64 de g = −4 en crudo (parcial); el control nuevo de 32 puntos (24 críticos + 8 al azar); procedimiento vs. azar (continuo); cómo sigue. 10 diapositivas. `figuras.py --grid <resultados>` y después `make_presentacion6_05_10.py`. |
| `presentacion7_10_10/` | 10/10/2026 | Avance de la reconstrucción con KMC para ir mostrando a los profes (estilo de la 5/6): estado de los 13 g; mapas finales de g = −8 a −5, −2 y +4 (KMC, continuo de referencia y la resta); control con 32 puntos (RMS crítico vs. al azar) y el ejemplo de g = −2; los 7 g que siguen eligiendo puntos; la grilla 64×64 de g = −4 completa; el techo del KMC con g > 0; KMC con checkpoint (corridas sin tope en g = +4); cómo sigue y qué consultar. 11 diapositivas. `figuras.py` (lee `autoexp/kmc/` traído de Mendieta) y después `make_presentacion7_10_10.py`. |
| `presentacion8_19_10/` | 19/10/2026 | Los errores del KMC explicados de cero, para la reunión del 19/10 (deck de claude.ai, no generador .py: fuente en `deck/`, link en el README de la carpeta). Ion, potencial μ y galvanostático (velocidad crucero, E = −μ); qué simula el KMC; error 1 (tasas de superficie con μ viejo → SoC ≈ 0 con Ξ alto) con dibujos de la superficie y el paso a paso; error 3 (el ion que salta se cuenta como su propio vecino → saltos × exp(−g/12)) con el dibujo del salto A → B; error 2 (fuera del arreglo → NaN); adelanto del 64×64 corregido; estado y preguntas. 16 diapositivas. |
| `explicacion_autoexp/` | 29/09/2026 | Material de estudio (no de reunión): el loop `autoexp` explicado paso a paso, con las notas del orador diciendo qué explicar en cada filmina. Reglas (oráculo/evaluador), muestreo en 3 etapas (grilla → bisección → relleno LOO), reconstrucción (borde RANSAC, cero abajo, rampa alineada, monotonía), chequeo de supuestos, fusión con DIP, resultados, KMC real y preguntas probables. Figuras calculadas con el código real (`figuras.py`, g = −2, 64 puntos). Agregado 3/10 (`figuras_kmc.py`): qué es la TPS (scipy `RBFInterpolator`) y por qué se llama "TPS con restricciones físicas", la receta en los 11 g del continuo con ruido (y g > 0 con más puntos), g = −4 KMC terminado con control, y dos secuencias con transición de fundido: el mapa tanda a tanda y los 8 pasos de la reconstrucción; y las 4 interpolaciones de scipy (vecino, lineal, cúbica, TPS) lado a lado. 40 diapositivas. `./venv/bin/python presentaciones/explicacion_autoexp/make_explicacion_autoexp.py` |
