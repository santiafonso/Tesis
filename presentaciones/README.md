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
