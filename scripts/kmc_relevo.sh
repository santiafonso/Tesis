#!/bin/bash
# Relevo de contenedores de Serafin para los loops kmc_auto (correr en el login de Mendieta).
# Uso: scripts/kmc_relevo.sh <actual>:<nuevo> [<actual>:<nuevo> ...]
#   p. ej. scripts/kmc_relevo.sh 431233:431939 431234:431940 431235:431941 431236:431942
# Para los loops, en cada g sin control.json cuyo contenedor vigente (reuse_next, o reuse_job si no
# hay) sea <actual>: reuse_job = <actual>, reuse_next = <nuevo>. Despues relanza con kmc_auto_all.sh.
# Los auto_args se leen solo al arrancar el loop: por eso hay que matarlos y relanzarlos.
cd "$(dirname "$0")/.." || exit 1
PY=${PYBIN:-$HOME/venv-autoexp/bin/python}
# pgrep con el patron partido para no matar este mismo script.
pids=$(pgrep -f "autoexp[.]kmc_auto --state")
[[ -n $pids ]] && kill $pids && sleep 3
$PY - "$@" <<'EOF'
import glob, os, sys
from autoexp import kmc_planner
m = dict(x.split(":") for x in sys.argv[1:])
for f in sorted(glob.glob("autoexp/kmc/g*/kmc.json")):
    if os.path.exists(os.path.join(os.path.dirname(f), "control.json")):
        continue
    st = kmc_planner.load(f)
    a = st.get("auto_args", {})
    cur = str(a.get("reuse_next") or a.get("reuse_job") or "")
    if cur not in m:
        print("%s: contenedor %s sin relevo, no se toca" % (f, cur))
        continue
    a["reuse_job"], a["reuse_next"] = int(cur), int(m[cur])
    kmc_planner.save(st, f)
    print("%s: %s -> %s" % (f, cur, m[cur]))
EOF
PYBIN=$PY scripts/kmc_auto_all.sh
