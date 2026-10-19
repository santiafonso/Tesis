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

## Usarlo con KMC real: `autoexp/kmc_planner.py`

```
python -m autoexp.kmc_planner init  --state kmc.json --budget 64 --sigma 0.01   # sigma: ruido de KMC (NRUNS)
python -m autoexp.kmc_planner next  --state kmc.json            # -> tanda_01.csv (row, col, logxi, logell, xi, ell)
#   ... correr KMC en esos puntos, armar resultados.csv con row,col,value (o logxi,logell,value) ...
python -m autoexp.kmc_planner add   --state kmc.json resultados.csv
python -m autoexp.kmc_planner next  --state kmc.json            # tanda siguiente, hasta "presupuesto completo"
python -m autoexp.kmc_planner reconstruct --state kmc.json --out mapa_kmc
python -m autoexp.kmc_planner simulate --g -2.0 --sigma 0.01    # prueba contra un mapa del continuo
```
Con 64 puntos son 9 tandas: 30 (grilla), 4-5 rondas de ~5 (bisección) y 3-4 de 4 (relleno).

En Serafín (`~/kmc/`: `kmc_fast` con gcc/12.3.0, `CS-40x20x40.xyz`, `kmc_tanda.slurm`), cada
tanda en su carpeta `~/kmc/tandas/g-4_tXX/` con `tanda.csv` y `logs/`:
```
sbatch ~/kmc/kmc_tanda.slurm tanda.csv     # 1 núcleo por corrida, NREP = 64 / puntos réplicas
python -m autoexp.kmc_results <carpeta traída> --out resultados.csv   # estado por rc (0/124/otro)
python -m autoexp.kmc_planner add --state autoexp/kmc/g-4/kmc.json resultados.csv
```
Estado del planificador para g=-4: `autoexp/kmc/g-4/` (kmc.json + tandas). El KMC siembra con
`srand(time(0))`: el slurm lanza de a una corrida por segundo para que las réplicas difieran.
Tiempo de corrida ~ 1/ℓ (piloto: 4 min en log ℓ=-0.5): log ℓ=-3.5 no entra en 48 h.

## KMC real: estado y cómo retomar (30/9)

- **g=-4 terminado**: 64 pts (56 medidos + 8 pseudo), mapa final `autoexp/kmc/g-4/dip/final.png`,
  control con 10 puntos no usados `control.{json,png}`: fusión TPS+DIP RMS 0.027 (máx 0.068) vs TPS
  0.043; el continuo usado como sustituto da 0.14. **Pendiente**: el job largo de Serafín
  (`kmc.json["largos"]`, puntos (102,10),(105,10) de log ℓ=-3.53) → al terminar, `kmc_results` +
  `kmc_planner add` (reemplazan los pseudo: hay que sacarlos de `pseudo` o forzar el valor) y rehacer
  `kmc_dip prepare` / DIP en Mendieta / `fuse` y `kmc_control eval`.
- **g=-3..4 en curso**: un loop por g (`autoexp/kmc/g<g>/`, `auto.log`), σ=0.005 (ruido medido entre
  réplicas 0.002–0.006). `scripts/kmc_auto_all.sh` relanza los pendientes (crontab @reboot). Opciones por
  g en `kmc.json["auto_args"]` (tag, race=false; g -3,-1,1,3 mandan los lentos a Serafín).
- **Desde 3/10: nodos reservados en Serafín.** Mulatona quedó inusable (4 nodos caídos) y la cola de
  Serafín es de ~1 día por job. Se pidieron 4 jobs `slurm/kmc/contenedor.slurm` (nodo entero, 48 h,
  `sleep`): 430560 = g0,g2 · 430561 = g4,g-2 · 430562 = g-1,g1 · 430563 = g-3,g3
  (`auto_args.reuse_job`, `reuse_cores` 32). Cada loop corre sus tandas como step adentro; las partes
  que esperaban en cola quedaron canceladas con `"hold": true` y `migrate()` las pasa al nodo cuando
  arranca. Si el nodo está en cola, el loop espera; si se termina (< 8.5 h), vuelve a mandar jobs
  normales a Serafín. **Al vencer (48 h) hay que pedir contenedores nuevos** (lo manda el usuario) y
  actualizar `reuse_job`.
- **Desde 4/10 los loops corren en el login de Mendieta**, no en la PC (`~/Tesis-autoexp`, con
  `PYBIN=~/venv-autoexp/bin/python scripts/kmc_auto_all.sh`; la clave de Mendieta está autorizada en
  Serafín, Mulatona y Mendieta). **El estado vivo (`autoexp/kmc/g*/kmc.json`, logs, tandas) está allá**:
  para traerlo, `rsync -a siaosorio@mendieta.ccad.unc.edu.ar:Tesis-autoexp/autoexp/kmc/ autoexp/kmc/`.
  Nunca correr loops en los dos lados a la vez (mandarían tandas dobles): el `@reboot` del crontab
  local tiene que estar comentado mientras corran en Mendieta.
- **4/10: pseudo = cota, relleno lejos de log l bajo.** (1) Los pseudo (KMC cortado por tiempo) son
  cotas inferiores: la reconstruccion usa `kmc_planner.recon_points` (valor = max(cota, TPS de lo medido),
  techo = maximo medido) + `lower` en `interp.cliff`. Antes entraban como valores y en g=+3 armaban un
  escalon falso a log Xi = -1.6 (cota 0.16 donde la rampa vale ~0.5). (2) `MIN_LOGELL_FILL = -3.3`: el
  relleno no elige log l < -3.3 (en g=-4 ni 48 h alcanzan en log l = -3.53; el relleno viejo mandaba casi
  todo a log l = -4). Se cancelaron las 7 tandas en vuelo que estaban ahi. g3 y g-4 se rehacen
  (`dip_v1/`, `control_v1.*` = lo viejo). Control extra en la zona izquierda de g3:
  `autoexp/kmc_control_zona.py` -> `autoexp/kmc/g3/control_zona.*`. (3) **La biseccion estaba apagada en
  g = -3, -2, 0, +1, +4**: el chequeo de forma la apagaba si la grilla violaba la monotonia, y en KMC la
  franja log Xi >= 1.4 da un poco menos que la de abajo (0.74 vs 0.98). Ahora solo la apaga un cero
  arriba/izquierda de un valor alto (`interp.zero_violations`), y no biseca columnas con log l < -3.3.
  (4) g1 y g4: presupuesto 96 (sin acantilado claro; en el continuo, 64 -> 96 suma 3-5 dB).
  (5) Una sola reconstruccion para todos los g: si `interp.cliff` no encuentra borde, ahora hace TPS +
  cotas + monotonia + techo (antes devolvia la TPS sola). g3 y g-4 no cambian; continuo g=+2/+4: +0.35 dB.
- **4/10 noche: presupuesto 96 en los 9 g** (en el continuo, 96 = 9/9 g >= 40 dB sin DIP, `runs/VAL9_96pts`;
  g3 con 64 no quedo bien). g-4 y g3 vuelven a correr (+32): lo de 64 quedo como `dip_64/`, `control_64.*`,
  `mapa_kmc_64.*`; `control.csv` se conserva (mismos puntos de control, sin KMC nuevo). Backup de los
  kmc.json previos: `autoexp/kmc/_backup_64_1004_1919/` (en Mendieta). Ojo: `pkill -f "autoexp.kmc_auto
  --state"` dentro de un `ssh '...'` mata la propia sesion (el patron esta en su linea de comando).
- **5/10: NODE_FAIL de los 4 contenedores** (4/10 20:50; Slurm los reencoló y volvieron 5/10 01:07,
  así que vencen 7/10 ~01:07). Los steps de la instancia vieja desaparecen de `sacct`: antes eso daba "?"
  (= red caída) y los loops esperaban para siempre (~16 h perdidas). Ahora `job_state` da `LOST` y
  `migrate()` relanza la tanda como step nuevo. g=-4 ya no usa Mulatona (`auto_args`: todo a Serafín,
  como step en 430561).
- **Mapa KMC denso 64×64 de g=-4** (pedido del profe, 5/10): `autoexp/kmc_grid.py`. 3392 puntos (log ℓ ≥ -3,
  las 11 columnas de abajo no entran en 48 h y quedan en gris), 1 réplica, 4 partes round-robin, ~3700
  núcleo-h estimadas. Serafín deja **4 jobs corriendo por usuario** (`AssocMaxJobsLimit`) y los 4 son los
  contenedores, así que la grilla corre **adentro de ellos con `nice -n 19`** (usa los núcleos que los loops
  dejan libres): parte k = `~/kmc/tandas/g-4_grid64_p{k}/` en 430560/61/62/63, lanzada con
  `srun --jobid=<cont> --overlap -N1 -n1 -c64 nice -n 19 bash ~/kmc/kmc_tanda.slurm tanda.csv` (reenviar
  retoma). Al final: `kmc_results` por parte + `kmc_grid mapa` → `autoexp/kmc/g-4/grid64/mapa.png`
  (crudo | continuo | diferencia). Sirve también de control denso para la reconstrucción de 96 puntos.
- **g = -8 … 4 de a 1** (pedido de los profes, 5/10): se sumaron g = -5, -6, -7, -8 con el mismo
  procedimiento (96 pts, σ 0.005), como steps de 16 núcleos: g-5 → 430560, g-6 → 430561, g-7 → 430562,
  g-8 → 430563 (y g-4 → 430561, 16 núcleos). Continuo de referencia: `GS_LIST="-5.0 -6.0 -7.0 -8.0"
  sbatch --array=0-3 slurm/phase_diagram_g.slurm` en Mendieta (job 1179537).
- **Control nuevo** (5/10): `kmc_control pick` elige 32 puntos, 3/4 sorteados con peso puntaje² donde
  puntaje = |rec − continuo| + |∇rec| (lo que más se aparta del continuo + el frente) y 1/4 uniformes;
  log ℓ ≥ -2.7. Deja `control_critico.png`. El control viejo de 10 puntos de g-4 y g3 quedó como
  `control_10.csv`, `resultados_control_10.csv` y `kmc.json["control_parts_10"]`.
- **Relevo de contenedores (5/10)**: pedidos 430773/74/75/76 (en cola por `AssocMaxJobsLimit`, arrancan
  cuando vencen 430560/61/62/63 el 7/10 ~01:07). Cada g tiene `auto_args.reuse_next`: `holder()` pasa solo
  al nuevo cuando corre, y mientras está en cola espera ("wait") en vez de mandar jobs sueltos. La grilla
  la releva `scripts/grid64_relevo.sh` (corriendo en el login de Mendieta, log en
  `autoexp/kmc/g-4/grid64/relevo.log`). Para el próximo relevo: pedir 4 más
  (`cd ~/kmc/contenedor && sbatch ~/kmc/contenedor.slurm` en Serafín), poner `reuse_next` y relanzar los loops.
- **Control: críticos vs al azar**: `kmc_control pick` escribe `control_tipos.json` y `eval` reporta
  `fusion_critico` / `fusion_uniforme` aparte en `control.json` (y los pinta distinto en `control.png`).
- **Procedimiento vs puntos al azar en el continuo**: `python -m autoexp.kmc_vs_random` → 
  `autoexp/kmc/comparacion_random/{psnr.csv,psnr.png,mapas.png}` (96 pts, σ 0.005, 10 semillas; el azar con
  nuestra reconstrucción y con TPS sola). g -8…-5 agregados 5/10 (merge a mano en psnr.csv).
  Resultado 5/10: el procedimiento gana +31-32 dB en g -8…-5, +28 en g=-4, +14-23 en g -3…-1 y g=1, +3-5 en g=0 y g ≥ 2;
  TPS sola sobre el azar da lo mismo que nuestra reconstrucción (±0.4 dB). Diapo en `presentacion4_autoexp`
  (`fig_vs_random`, lee `psnr.csv` + `ejemplos.npz`).
- **Clusters**: Mulatona desaloja (partición `batch` de la cuenta `iate` pisa `mono`/`short`); Serafín no
  (nodo entero, cola larga). Corridas de log ℓ ≤ -3 tardan 1.5–3 días: solo en Serafín, o pseudo.
  Un job propio corriendo se puede reusar con `srun --jobid=<id> --overlap` (`--reuse-job`).
- **7/10: los puntos medidos mandan** (`interp.cliff(honor=tol)`, `kmc_planner.recon_kw` lo pone en
  max(2σ, 0.01)). El KMC no es monótono en Ξ (franja de log Ξ alto más baja) y las cotas + proyección monótona
  usaban esos puntos como techo de toda la columna: en g=-8 la TPS erraba 28/75 puntos medidos por > 0.1
  (0.98 → 0.67) y el control daba RMS 0.30. Ahora los pares contradictorios no ponen cotas y al final se suma
  la TPS de los residuos (`interp.honor_points`, también sobre el final de `kmc_dip fuse`). Control TPS:
  g=-8 0.296 → 0.017, g=-4 (10 pts) 0.033 → 0.008, g=3 0.006 → 0.012. g=-8 rehecho (lo viejo: `dip_v1/`,
  `control_v1.*`, `mapa_kmc_v1.npy`). Solo afecta al KMC (sin `honor` el continuo queda igual).
  OJO: el loop termina cuando el planificador (re-jugado con las reglas actuales) no pide más, no al llegar a
  96 conocidos: los g con puntos de antes del 4/10 piden algunos más (g=-2: ~19). El usuario dijo que está bien.
- **7/10: relleno sin la franja de log Ξ alto.** En g=-8 la grilla (30) y la bisección (hasta 49) andaban bien,
  pero el relleno LOO mandó 30 de sus 47 puntos a log Ξ ≥ 1.3: ahí el KMC da corridas "sospechosas" (terminan
  rc=0 en 4-10 min con SoC ≈ 0 y E final 0.25-0.37, contra ~1-2 h, 0.99 y E 0.05-0.10 de las normales) y
  réplicas bimodales (0.99/0.99/0.99/0.00), así que el error LOO ahí nunca baja. Ahora: `MAX_LOGXI_FILL = 1.4`
  (`min_row` en `sampling.loo_fill`: ni se elige ni aporta error LOO), los pseudo no aportan error LOO (`skip`),
  y `kmc_results` usa la mediana de las réplicas + columna `bimodal`. Los 12 loops se relanzaron (9:36); g=-8
  queda fuera (DIP en curso). **Pendiente: preguntar a los profes por qué pasa** (lista en
  `autoexp/kmc/pseudo_por_g.csv`, figura `autoexp/kmc/g-8/pseudo.png`).
- **8/10: bimodales viejos recargados con la mediana.** 12 puntos cargados antes del 7/10 tenian el promedio
  de replicas bimodales (p. ej. 0.98/0.99/0.99/0 -> 0.74): g -8 (3), -7, -5 (3), -3 (2), -2 (2), 4. Arreglados
  en los kmc.json de Mendieta con `autoexp/kmc/fix_bimodal.py` (alla, sin commitear). g=-7 y g=-8 rehacen
  el final (lo viejo: `dip_pre_mediana/`, `mapa_kmc_pre_mediana.*`, `control_pre_mediana.*`).
- **8/10: mapa 64x64 de g=-4 armado**: `autoexp/kmc/g-4/grid64/mapa.png` (copia liviana en `crudo/`, en
  Serafin `~/kmc/grid64_liviano/`). 3392 puntos, RMS vs continuo 0.234. La franja log Xi >~ 1.3 con
  log l < -1.5 da SoC ~ 0 entera (no puntos sueltos), el acantilado esta corrido a Xi bajo y hay
  pixeles sueltos en ~0 dentro de la zona de SoC alto. Pendiente revisarlo con el usuario.
  Figura "cruda" para los profes (solo terminadas, sin continuo): `grid64/mapa_crudo.png`.
  **Resto de la grilla** (pedido del usuario, tope 24 h por corrida): columnas log l < -3 + los 40
  cortados (a las 36 h por fin de contenedor, no por lentos), 744 puntos en `grid64/resto/parte_{1..4}.csv`
  -> Serafin `~/kmc/tandas/g-4_grid64_resto_p{k}/`. `scripts/grid64_resto.sh` (login de Mendieta) las
  lanza en los contenedores 431233-36 cuando arrancan (9/10 ~01:07). Tiempo por corrida (nucleo propio)
  ~ 8x por unidad de log l: -3.05 ~22 h, -3.5 ~59 h, -4 ~157 h; con nice 19 ~2x. Lo de log l < ~-3.1
  sale como cota a las 24 h. Para terminarlas haria falta checkpoint en el KMC (Serafin/Mulatona/Mendieta
  cortan a 48 h; la `batch` de 7 d de Mulatona es solo cuenta `iate`).
- **10/10: KMC con checkpoint, corridas sin tope** (para ver si el techo bajo de g=+4, SoC max medido 0.86 vs
  ~1 del continuo, es real o efecto de las cotas en log l < -2.5). `KMC_CKPT=1` en el entorno -> guarda el
  estado completo (red, generador, tasas y Fenwick) en `ckpt-*.bin` cada `KMC_CKPT_EVERY` s (600) y al
  SIGTERM; relanzar en la misma carpeta retoma **bit a bit** igual (probado: md5 identico cortando 2 veces y
  contra el binario viejo). Sin la variable el binario es el de siempre. En Serafin `~/kmc/kmc_ckpt`
  (fuente en `~/kmc/ckpt_src/`), step `slurm/kmc/kmc_largo.sh` (no borra datos si hay checkpoint) y
  `scripts/kmc_largo_relevo.sh` (login de Mendieta) lo pasa de contenedor en contenedor. Tanda: 6 puntos
  g=4 x 2 replicas (`autoexp/kmc/g4/largos_ckpt/tanda.csv`, Serafin `~/kmc/tandas/g4_largos_ckpt/`):
  5 de las cotas (log l -3.53 a -2.54) + el punto medido mas alto (0.859, log l -2.02) como referencia.
  Lanzado 10/10 18:00 en 431234 -> 431940 (log `autoexp/kmc/g4/largos_ckpt/relevo.log` en Mendieta).
  log l -3.53 tarda ~2.5 d: para los contenedores siguientes relanzar el relevo con los IDs nuevos.
- Estado de todo: `for d in autoexp/kmc/g*/; do tail -2 $d/auto.log; done`; figuras por g:
  `estado.png`, `avance.png`, y al final `dip/final.png`, `control.png`.
- Anomalía del KMC para consultar: log Ξ ≥ 1.4 y log ℓ ≤ -3.5 → SoC ≈ 0 en ~3 min (continuo ~1),
  reproducible (6/6 semillas).
- **Presentación siguiente (`presentaciones/presentacion6_05_10/`)**: los g negativos se muestran SIEMPRE con
  KMC (el continuo solo como referencia, o donde se mide el método contra imagen completa). Para actualizarla:
  copiar `autoexp/kmc/g*/kmc.json` desde Mendieta; para la grilla, en Serafín armar una copia liviana
  (`estado.txt` + `tail -5` de cada `datos-*.dat`, ~0.5 MB; las carpetas pesan 750 MB por parte), bajarla con
  `rsync -e "ssh -J siaosorio@mendieta.ccad.unc.edu.ar" siaosorio@serafin.ccad.unc.edu.ar:...`, `kmc_results`
  por parte, y `figuras.py --grid <carpeta>` + `make_presentacion6_05_10.py`. `kmc_grid mapa` ya no mete las
  cotas (corridas en curso) en la diferencia ni en el RMS.
- **12/10: transectas sobre la tendencia KMC − continuo** (pedido del usuario: "tirá puntos donde va la
  tendencia de la diff para ver si sigue en la misma dirección"). Tendencia medida en los mapas finales: el
  acantilado del KMC está corrido en log Ξ ≈ 0.087·g + 0.05 (−0.85 en g=−8, +0.6 en g=+4) y, para g<0, la
  caída en ℓ se adelanta (−1.2 en g=−8, ~0 en g≥−2). 230 puntos (1 réplica): cruzan el acantilado en
  log ℓ = −2, −1, 0 y, para g≤−2, la caída en log Ξ = 1, 0, desde el continuo hasta lo extrapolado ±0.25.
  `autoexp/kmc/tendencia/` (tanda.csv, p1-4.csv, tendencia_pts.py; todos_finales.py = figura de los 13 g
  con continuo, diferencia y puntos de control). Corren como steps en los contenedores 431939/41/42/40
  (Serafín `~/kmc/tandas/tendencia_p{1..4}/`), tope 10 h; los contenedores terminan 13/10 ~01:15.
  NO se cargan en los kmc.json de los loops: analizarlos aparte (`kmc_results` por parte).
  Verificado: todas las tandas corrieron con su propio g (estado.txt), misma fuente en Serafín y Mulatona.
  scp a Serafín se cuelga: copiar con `ssh host "cat > archivo" < archivo`.
- **12/10: bug en `kmc_results` (filas basura al final de `datos-*.dat`).** Algunas corridas siguen escribiendo
  después del final: 2-6 filas con SoC 0 y E 0 o -inf (y t creciente). Se descartan las -inf, pero la fila con
  E = 0.000 se toma como última → SoC 0. Ejemplos de la grilla 64×64: r006 c025 da 0 y en realidad es 0.90; r022 c022
  da 0 y es 0.97; r014 c027 da 0 y es 0.93 (son los 7 píxeles sueltos en 0 dentro de la zona amarilla). En los loops: g-1 (13,23)
  quedó en 0.0 por esto (1 sola réplica "terminada"). Afectadas: g -8 (2 pts), -5 (4), -3 (3), -2 (1), -1 (2);
  casi todas tapadas por la mediana. Arreglo pendiente: tomar el SoC máx de las filas válidas.
  **g-1: loop detenido y DIP 1180008 cancelado (12/10)** hasta corregir esos puntos. Además (14,21), (16,18),
  (13,26) de g-1, (15,20) de g-5 y (16,18), (19,15) de g3 tienen una réplica "terminada" por debajo de la cota de
  las cortadas (0.25 contra 0.33-0.5): son bimodales en la zona anómala → usar la cota.
  Grilla 64×64 completa: `grid64/final_1012/mapa.png` (4096 puntos, 499 cotas a 24 h en log ℓ < -2.9).
- **12/10 tarde: lector corregido y aplicado.** `kmc_results`: (1) saca las filas basura del final (SoC 0 con E 0);
  (2) si las réplicas cortadas ya superan la mediana de las terminadas por > 0.05 → el punto queda como cota.
  Re-leídas todas las tandas (`~/.cache/kmc_auto` en Mendieta, `~/reeval/`): 33 cambios reales en g -8,-7,-6,-5,
  -3,-2,-1,+3 (casi todos +0.01-0.07; grandes: g-1 4 pts, g-5 (15,20), g3 (19,15),(16,18) → cotas; g-8 hasta +0.13).
  Backups `kmc_pre_1012.json`. g-1 (13,23) (1 réplica terminada en 0.25, 7 colapsadas) → cota a mano.
  Relanzados los loops de g -1, -5, -8, +3 (rehacen DIP final + eval del control con los mismos puntos de control;
  lo viejo en `dip_pre_1012/`, `control_pre_1012.*`; g-1 roto en `dip_roto_1012/`).
- **12/10: grilla 64×64 hasta el final, con checkpoint.** Las 499 cotas (log ℓ ≤ -2.86) se corren de cero con
  `kmc_ckpt` (1 réplica, `NREP=1`): Serafín `~/kmc/tandas/g-4_grid64_ckpt_p{1..4}/` (partes en
  `grid64/ckpt/parte_k.csv`, ~8300 núcleo-h cada una, la más larga ~6.6 d). Un relevo por parte en el login de
  Mendieta (`scripts/kmc_largo_relevo.sh`, ahora con `NREP`), logs `grid64/ckpt/relevo_p{k}.log`. Arrancaron
  en 431939/41/42/40 (vencen 13/10 ~01:15). **Cada 48 h: pedir 4 contenedores nuevos (el usuario) y relanzar
  los 4 relevos con los IDs nuevos** (`NREP=1 setsid nohup scripts/kmc_largo_relevo.sh <dir> 64 "<id>"`).
  Comparación reconstrucción de 96 pts vs crudo: `grid64/vs_reconstruccion.py <dir> final.npy kmc.json`.
- **12/10: relevo a 432112-15** (pedidos por el usuario). `kmc_relevo.sh 431939:432112 431940:432113 431941:432114
  431942:432115 431233:432112 430776:432113` + relevos de la grilla (`NREP=1`) y de `g4_largos_ckpt` con los IDs
  nuevos. Los jobs sueltos en cola (controles 431953/55 y 432108-10 de g-5/g-8) quedaron con `scontrol hold`
  para que los contenedores arranquen primero; los loops los migran como steps (migrate ve PENDING).
  Si algo no migra: `scontrol release <id>`.
- **Figura del crudo para los profes**: `grid64/figura_profes.py` → `grid64/final_1012/crudo_profes.png` (KMC
  64×64 con la zona cortada a 24 h rayada, borde del continuo, línea de tendencia + transectas; y KMC − continuo).
  Corrige 9 puntos del bug de filas basura (`correccion_filas_basura_1012.txt`, leído de los datos completos en
  Serafín). La zona de SoC ≈ 0 arriba a la izquierda es real (no es el bug).
- **13/10: mandado a los profes (WhatsApp) el crudo 64×64 `grid64/final_lector/crudo_solo.png`** (lector
  corregido; el de `final_1012/` nunca se mandó). Se les dijo: SoC=0 arriba del gráfico, patrón en todos los g
  (el punto log Ξ 1.43 / log ℓ -3.53 es "sospechoso" en los 13 g, `pseudo_por_g.csv`), y que la zona rayada
  se termina con checkpoint para tener el 100 % terminado. Preguntas abiertas: (1) zona log Ξ ≳ 1.3, ℓ bajo: termina en minutos sin intercalar (SoC ~0, E final ~0.28 V), continuo ~1,
  bimodal según semilla en el borde; (2) franja log Ξ 0.9–1.8: la adsorción no sigue la corriente impuesta
  (SoC 10–15 % atrasado, termina por fin del tiempo nominal sin tocar el corte) → salpicado verde con 1 réplica;
  (3) ¿está bien terminar por tiempo nominal y no por corte? Pendiente según respuesta: 4 réplicas en esa franja
  (11–18 h por corrida) o mirar tasas/paso de tiempo en el código.
- **13/10: causa del SoC ≈ 0 arriba del mapa → `autoexp/kmc/investigacion_soc0/NOTAS.md`.** Al cambiar `mui`
  solo se actualizan las tasas de ads/des del sitio tocado + vecinos (el resto de la superficie queda con `mui`
  viejo). Con refresco de toda la superficie, el punto log Ξ 2 / log ℓ -1.33 carga normal; un punto normal no
  cambia. Las tasas viejas están también en la versión vieja sin optimizar, pero ahí carga bien: la optimizada
  calcula las tasas iniciales con el `mui` resuelto (E ≈ +0.35 V) y la vieja con `mui = 0`, y los sitios que
  nunca se tocan quedan congelados en ese valor → el SoC ≈ 0 lo dispara el orden de inicialización de la
  optimizada (14/10, detalle en NOTAS.md). Corridas en Serafín `~/kmc/investigacion_soc0/`. Pendiente:
  decidir con los profes si se agrega `RefrescarSuperficie()` (2-20× más lento) y se rehace la zona. Los profes respondieron por audio (13/10): "debe ser un error",
  "fijate si son NaN" (no lo son), y que en extremos el dt puede no integrar bien saltos de potencial.
- **14/10: segundo bug del KMC, escritura fuera del arreglo (de acá salen TODOS los NaN).** El código no tiene fin por
  tiempo: si la corrida pasa el tiempo nominal (200 intervalos, `Nbins`) sin tocar el corte, l.337-348 escriben en
  `MUs/tita[Time_i >= 200]`, pisan otras variables y a los ~6 intervalos `mui` = NaN → el `while (mui < muoff)` sale con
  rc=0 (filas basura del final). 41 de 67 `datos` completos locales terminan así; ningún NaN dentro de los 200. No causa
  el SoC ≈ 0 (ya era 0 antes). El continuo (`galpynostatic/lib/map.cpp`, `while (pot_i > vcut)`) corre hasta el corte
  sin tope → propuesta: agrandar arreglos + tope `NbinsMax`. Ojo: la basura puede parecer válida (SoC 0.085, E -0.18).
  Tanda con `RefrescarSuperficie()` (40 pts, `~/kmc/tandas/g-4_refresco_{a,b}`) terminada: zona SoC 0 → 0.97; log Ξ ≤ 1.1
  cambia ≤ 0.04; log Ξ alto / log ℓ -0.5 baja 0.03-0.08; vs reconstrucción 96 pts RMS 0.029 (64 pts: 0.033). La caída
  adelantada en ℓ (log ℓ 0: 0.40 vs 0.66 continuo) sigue igual con el arreglo; el corrimiento del acantilado en Ξ no se midió.
  Resumen para la reunión del 15/10: `autoexp/kmc/investigacion_soc0/para_profes.txt`.
- **16/10: crudo 64×64 de g=-4 con el código corregido** (`RefrescarSuperficie()`, binario `~/kmc/refresco/kmc_ckpt`
  enlazado como `kmc_fast`; el fin por NaN NO está corregido, igual que el crudo viejo, para comparar solo el error 1).
  log ℓ ≥ -2.5 (48 columnas, 3072 pts, 1 réplica; ℓ más bajo tardaría semanas), estimado ~5600 núcleo-h, corrida más
  larga ~34 h. Tandas `autoexp/kmc/g-4/grid64_refresco/parte_{1..4}.csv` → Serafín `~/kmc/tandas/g-4_grid64_refresco_p{k}/`,
  jobs sueltos 433568-71 (`sbatch --time=2-00:00:00 --export=ALL,KMC_DIR=$HOME/kmc/refresco ~/kmc/kmc_tanda.slurm`).
  Al terminar: `kmc_results` por parte + `kmc_grid mapa` y comparar contra `grid64/final_lector/mapa_corregido.npy`.
  **Plan (usuario, 16/10):** ver primero el crudo corregido vs el viejo y recién en base a eso decidir cómo seguir con las
  reconstrucciones de los demás g (rehacer con el binario corregido o no). Los loops de g0 y g-4 seguían vivos en Mendieta
  (presupuesto completo 12/10, en la etapa final DIP/control, con el binario viejo).
- **16/10: prueba g=+4 con el binario corregido** (¿el techo 0.86 / el verde de g=+4 es el bug?). 12 puntos medidos de
  g4 × 5 réplicas (nodo entero obligatorio en Serafín: ≥ 60 CPUs): 6 en la meseta (log Ξ ≥ 0.2, KMC 0.78-0.86 vs continuo
  0.94-0.97) y 6 en el acantilado (log Ξ -1.9 a -0.6, KMC 0.2-0.3 por debajo del continuo). `autoexp/kmc/g4/refresco_prueba/tanda.csv`
  → Serafín `~/kmc/tandas/g4_refresco_prueba/`, job 433574 (con 4 jobs/usuario arranca cuando termine una parte del crudo).
  Comparar contra `kmc.json["known"]` de g4 en Mendieta. Diagnóstico previo: en g<0 el verde está en log Ξ ≥ 0.9 con KMC
  0.2-0.4 debajo del continuo (zona del bug); en g=+4 el verde está en todo el mapa y el SoC máx medido es 0.86.
- **16/10: Maxi confirmó el arreglo del error 1** (WhatsApp, 15:01): "idealmente la actualización siempre es local, a menos
  que necesite recalcular el potencial, que ahí lo hace para todos los sitios de la superficie" / "para el potencial
  necesitás todas las tasas". Es lo que hace `RefrescarSuperficie()` (solo dentro del `if (g_need_potencial)`). La reunión
  pasó al lunes 19/10: llevar el crudo 64×64 corregido vs el viejo. Captura del código: `investigacion_soc0/bug1_codigo.png`.
- **16/10: tercer error, confirmado por Maxi (15:10): `CalcularEf` en `Actualizar` (l.628).** La energía del sitio destino
  cuenta como vecino al propio ion que salta (el origen sigue `Ocup = 1`; `numvec` = `numven`) → J1 de más en todos los
  saltos → difusión × exp(−g/12) (g=−8 ×1.95, g=−4 ×1.40, g=+4 ×0.72). También en la versión vieja (l.888). Arreglo:
  `dE = (CalcularEf(i, jv) - J1) - Esitio`. El crudo corregido que corre (433568-71) y la prueba g4 (433574) NO lo tienen.
- **16/10 15:44: binario con los dos arreglos en uso.** Serafín `~/kmc/refresco/kmc_fast -> kmc_ckpt_ef` (fuente
  `KMC-Galvanostatic_noclus_param_claude_ef.cpp` = checkpoint + `RefrescarSuperficie()` + `- J1` en l.638; local
  `investigacion_soc0/kmc_ckpt_refresco_ef.cpp`). El de solo refresco quedó como `kmc_ckpt`. Los jobs en cola 433568-71
  (crudo) y 433574 (g4) arrancan con este. Probado local (`investigacion_soc0/run_ef_*`, -DFIXEDSEED): el punto anómalo
  carga como con solo refresco, el normal igual. Loops de g0 y g-4 en Mendieta detenidos (binario viejo).
- **18/10: adelanto local del crudo con los dos arreglos (24 pts g=-4)**: `kmc/investigacion_soc0/comparacion_ef/`
  (log Ξ 2…0 × log ℓ -1.05, -0.48, 0; binario `kmc_ef_rand`, 1 réplica). Figuras `comparacion_soc.png` (viejo / solo
  superficie / ambos / continuo) y `curvas_E_soc.png`. log ℓ -1.05 casi igual; con `- J1` el SoC baja en log ℓ -0.48
  (~0.80 → 0.67) y 0 (~0.41 → 0.31): la difusión queda más lenta (antes ×1.40 en g=-4) y el KMC se aleja más del
  continuo (0.88 / 0.66). En `curvas_E_soc.png` el "viejo" y "solo superficie" del punto normal son corridas locales
  cortadas (rc=143), no terminan.
- **19/10 (mañana): adelanto del crudo corregido** (433568-71 arrancaron 19/10 04:13-07:22, ~45 h): 153 terminados +
  256 en curso, todos en log ℓ -2.48…-1.90. `autoexp/kmc/g-4/grid64_refresco/adelanto_1910/adelanto.png` (+ `adelanto.py`,
  `res_p*.csv` de una copia liviana). La zona de SoC 0 arriba (log Ξ ≥ 1.8) pasa a 0.87-0.97; el acantilado en Ξ no se
  mueve; el resto cambia ≤ 0.09. RMS vs continuo en los 48 no triviales: 0.373 viejo → 0.238 corregido.
- **19/10 10:17: reconstrucción de los 13 g relanzada con el KMC corregido** (pedido del usuario, sin esperar el crudo).
  Estados nuevos en `autoexp/kmc_ef/g<g>/` (en Mendieta; los viejos de `autoexp/kmc/` quedan intactos): 96 pts, σ 0.005,
  mismo muestreo, tag `g<g>_ef`, todo a Serafín, `auto_args.kmc_dir = $HOME/kmc/refresco` (binario `kmc_ckpt_ef`: refresco de
  superficie + `- J1`; el fin por NaN después de 200 intervalos sigue sin corregir). `kmc_auto` ahora pasa `KMC_DIR` a
  `kmc_tanda.slurm` (opción `kmc_dir`/`--kmc-dir`) y `scripts/kmc_auto_all.sh` toma `KMC_ROOT` (lanzado con
  `KMC_ROOT=autoexp/kmc_ef PYBIN=~/venv-autoexp/bin/python`). Tanda 1 = 26 jobs sueltos (433731-54) en cola por
  `AssocMaxJobsLimit` detrás del crudo corregido (433568-71, ~45 h). Sin contenedores: cuando el usuario los pida, poner
  `reuse_job` (y `reuse_cores` 16) en los `auto_args` y relanzar los loops para que migren lo pendiente.
- Figura de los 13 g regenerada con el estado de Mendieta del 19/10: `autoexp/kmc/tendencia/todos_finales.png` (copia de los
  finales en `tendencia/kmc/`; la del 12/10 quedó como `todos_finales_1210.png`).
- **19/10 10:40: contenedores 433757-60** (pedidos por el usuario, en cola detrás del crudo). `reuse_job`/`reuse_cores 16`:
  433757 = g -8,-4,0,4 · 433758 = g -7,-3,1 · 433759 = g -6,-2,2 · 433760 = g -5,-1,3. Los 26 jobs sueltos de la tanda 1
  (433731-56) quedaron con `scontrol hold` para que arranquen primero los contenedores; los loops (relanzados) los migran
  como steps. 433574 (prueba g4) sigue en cola normal y va a tomar uno de los 4 lugares.
- **19/10: el fin por NaN (error 2) sigue con el binario corregido.** En el adelanto del crudo corregido, 7 de 153 corridas
  terminadas llegan al tiempo nominal sin tocar el corte y terminan por NaN: todas en log ℓ -2 con log Ξ alto (2, 1.62,
  1.05...), con SoC 0.87-0.97 y E +0.01 a +0.04 V en el último intervalo válido (corte -0.149). Las normales terminan
  por corte en el intervalo que no se imprime. Preguntas para los profes en la sesión del 19/10.
