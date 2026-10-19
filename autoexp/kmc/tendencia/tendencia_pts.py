"""Transectas sobre la tendencia KMC - continuo: cruzan el acantilado (en log l = -2, -1, 0) y, para g<0,
la caida en l (en log Xi = 1, 0), desde la posicion del continuo hasta la extrapolada por la tendencia en g,
+-0.25 de margen. Salida: tendencia/tanda.csv (row,col,logxi,logell,xi,ell,g)."""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
from autoexp.oracle import load_truth
S = os.path.dirname(os.path.abspath(__file__))
lx = np.linspace(2, -4, 128); le = np.linspace(-4, 2, 128)
GS = list(range(-8, 5))
def cliff(img, c):
    col = img[::-1, c]; i = np.argmax(col > 0.5); return lx[::-1][i] if col[i] > 0.5 else np.nan
def fall(img, r):
    idx = np.where(img[r] > 0.5)[0]; return le[idx.max()] if len(idx) else np.nan
LS, XS = (-2.0, -1.0, 0.0), (1.0, 0.0)
cols = [int(np.argmin(abs(le - v))) for v in LS]; rows = [int(np.argmin(abs(lx - v))) for v in XS]
dc, df = {}, {}
for g in GS:
    p = f'{S}/kmc/g{g}/dip/final.npy'
    if not os.path.exists(p): continue
    k = np.clip(np.load(p), 0, 1); c = load_truth("%.1f" % g)
    dc[g] = np.nanmedian([cliff(k, j) - cliff(c, j) for j in cols])
    df[g] = np.nanmedian([fall(k, i) - fall(c, i) for i in rows])
ga = np.array(sorted(dc)); pc = np.polyfit(ga, [dc[g] for g in ga], 1)
gn = np.array([g for g in ga if g <= 0]); pf = np.polyfit(gn, [df[g] for g in gn], 2)
print("tendencia acantilado dXi = %.3f g %+.3f ; caida dl (g<=0) = %.4f g^2 %+.3f g %+.3f" % (*pc, *pf))
out = []
for g in GS:
    st = json.load(open(f'{S}/kmc/g{g}/kmc.json'))
    known = np.array([list(map(int, s.split(','))) for s in st['known']])
    c = load_truth("%.1f" % g)
    def add(r, cc):
        r = int(np.clip(r, 0, 127)); cc = int(np.clip(cc, 0, 127))
        if np.min(np.abs(known - [r, cc]).sum(1)) <= 1: return
        if any(o[0] == r and o[1] == cc and o[6] == g for o in out): return
        out.append((r, cc, lx[r], le[cc], 10 ** lx[r], 10 ** le[cc], g))
    sh = np.polyval(pc, g)
    for j in cols:
        a = cliff(c, j); b = a + sh
        for v in np.linspace(min(a, b) - 0.25, max(a, b) + 0.25, 5):
            add(np.argmin(abs(lx - v)), j)
    if g <= -2:
        shf = np.polyval(pf, g)
        for i in rows:
            a = fall(c, i); b = a + shf
            for v in np.linspace(min(a, b) - 0.25, max(a, b) + 0.25, 5):
                if v >= -2.1: add(i, np.argmin(abs(le - v)))
os.makedirs(f'{S}/tendencia', exist_ok=True)
with open(f'{S}/tendencia/tanda.csv', 'w') as f:
    f.write("row,col,logxi,logell,xi,ell,g\n")
    for o in out: f.write("%d,%d,%.6f,%.6f,%.8g,%.8g,%d\n" % o)
print(len(out), "puntos;", {g: sum(o[6] == g for o in out) for g in GS})
