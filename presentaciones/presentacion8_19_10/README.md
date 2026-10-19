# presentacion8_19_10 — Errores del KMC (reunión 19/10/2026)

Deck: https://claude.ai/artifact/Qzd1TnWtQjA8J7nu7FbL99 (privado; compartir desde Share, y desde ahí
Export → .pptx / PDF). El .pptx exportado está acá: `presentacion8_19_10.pptx`.

- `deck/` — fuente del deck (`deck.json` + una `slides/<id>.html` por filmina, con notas del orador en `<aside>`).
- `fig/` — figuras usadas o de respaldo:
  - `adelanto.png` — crudo 64×64 de g=−4 corregido vs viejo (153 pts, 19/10; `autoexp/kmc/g-4/grid64_refresco/adelanto_1910/adelanto.py`).
  - `comparacion_soc.png` — 24 puntos g=−4: viejo / solo arreglo 1 / arreglos 1+3 / continuo (`autoexp/kmc/investigacion_soc0/comparacion_ef/`).
  - `todos_finales.png` — los 13 g reconstruidos con el binario viejo, estado de Mendieta al 19/10 (`autoexp/kmc/tendencia/todos_finales.py`).

Contenido (17 filminas): ion / potencial μ / galvanostático como velocidad crucero (E = −μ: μ más alto → entran
más iones); qué simula el KMC (adsorción, desorción, salto); error 1 (tasas de superficie con μ viejo → SoC ≈ 0 con
Ξ alto; arreglo `RefrescarSuperficie()`); error 3 (`CalcularEf` cuenta al ion que salta como vecino de su destino →
saltos × exp(−g/12); arreglo `− J1`); tiempo nominal (capacidad / corriente, C-rate, 7 de 153 corridas corregidas lo pasan sin tocar el corte); error 2 (escritura fuera del arreglo después de 200 intervalos → NaN; arreglo
pendiente); estado y preguntas.
