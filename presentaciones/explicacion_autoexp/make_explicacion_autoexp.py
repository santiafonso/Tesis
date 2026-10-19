#!/usr/bin/env python
# coding: utf-8
"""Genera explicacion_autoexp.pptx: el loop autoexp explicado paso a paso, para estudiarlo
(no es la presentacion de resultados: esa es presentacion4_autoexp). Cada filmina trae en las
NOTAS del orador que decir.

Las figuras didacticas se calculan con el codigo real (figuras.py, caso g=-2.0 con 64 puntos);
el recorrido de puntajes sale de autoexp/runs/*/score.json via el generador de la 4; los numeros
de validacion en 9 g, ruido y KMC vienen de autoexp/experimentos.md y estan escritos aca.

Uso (desde ~/Tesis-autoexp):
    ./venv/bin/python presentaciones/explicacion_autoexp/make_explicacion_autoexp.py
"""
import os
import sys

import numpy as np

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "fig")
FIG4 = os.path.join(REPO, "presentaciones", "presentacion4_autoexp", "fig")
OUT = os.path.join(HERE, "explicacion_autoexp.pptx")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "presentaciones", "presentacion4_autoexp"))

import figuras  # noqa: E402
import figuras_kmc  # noqa: E402
from make_presentacion4_autoexp import STEPS, dev_stats  # noqa: E402

# g > 0 (sin borde): que pasa con mas puntos (scratch presupuesto.py, sigma 0.005, interp.auto)
PRESUPUESTO_TXT = ("No es que el corte empeore: no se activa. Con 96 puntos en vez de 64: "
                   "g=+2 35.3 → 38.0 dB, g=+4 36.8 → 42.0 dB. Ahí conviene medir más.")
PRESUPUESTO_NOTA = """¿Conviene tirar más puntos donde no hay borde? Sí. Mismo planificador, σ = 0.005, interp.auto:
g = +2: 64 puntos 35.3 dB -> 96: 38.0 -> 128: 41.4.  g = +4: 64 -> 36.8, 96 -> 42.0, 128 -> 42.6.
Sin borde, la información está repartida en toda la rampa (no concentrada en un salto), así que cada
punto extra rinde. Regla práctica para KMC: si después de la grilla y la bisección no aparece borde,
subir el presupuesto (~96) en vez de insistir con 64."""


def rgb(h):
    return RGBColor.from_string(h)


NAVY, ORANGE, BLUE, GREEN = "14213D", "EB6834", "2A78D6", "1BAF7A"
INK, MUTED, WHITE = "1F2328", "5B6270", "FFFFFF"
CODEBG, BLUE_T, ORANGE_T, GREEN_T = "F2F4F7", "EAF2FC", "FDEEE7", "E6F6EF"
FONT, MONO = "Calibri", "Courier New"
SW, SH = 13.333, 7.5


# ---------------------------------------------------------------- helpers ------
def new_slide(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(NAVY if dark else WHITE)
    return s


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text.strip()


def _run(p, text, size, bold=False, color=INK, font=FONT, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = rgb(color)
    return r


def text(s, x, y, w, h, paras, size=16, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         font=FONT, space=6, bullets=False):
    """paras: lista de str o de listas de (texto, {bold, color, size}) para mezclar estilos."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for k, para in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        if bullets:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.28))))
            pPr.set("indent", str(-int(Inches(0.28))))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            pPr.append(bu)
        for chunk in ([(para, {})] if isinstance(para, str) else para):
            t, st = chunk
            _run(p, t, st.get("size", size), st.get("bold", False), st.get("color", color),
                 st.get("font", font), st.get("italic", False))
    return tb


def box(s, x, y, w, h, fill, radius=True):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                            Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    sh.line.fill.background()
    sh.shadow.inherit = False
    if radius:
        sh.adjustments[0] = 0.06
    return sh


def title(s, t, step=None, dark=False, sub=None):
    x = 0.6
    if step is not None:
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(0.5), Inches(0.62), Inches(0.62))
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(ORANGE)
        c.line.fill.background()
        c.shadow.inherit = False
        tf = c.text_frame
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _run(tf.paragraphs[0], str(step), 20, True, WHITE)
        x = 1.4
    text(s, x, 0.42, SW - x - 0.6, 0.8, [t], size=32, color=WHITE if dark else NAVY,
         anchor=MSO_ANCHOR.MIDDLE, space=0)
    s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    if sub:
        text(s, x, 1.18, SW - x - 0.6, 0.5, [sub], size=16, color=MUTED, space=0)


def image(s, path, x, y, w=None, h=None):
    """Imagen dentro de la caja (x, y, w, h) respetando la proporcion, centrada."""
    iw, ih = Image.open(path).size
    r = iw / ih
    if w and h:
        if w / h > r:
            nw, nh = h * r, h
        else:
            nw, nh = w, w / r
        x, y = x + (w - nw) / 2, y + (h - nh) / 2
        w, h = nw, nh
    elif w:
        h = w / r
    else:
        w = h * r
    return s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def code(s, x, y, w, h, src, size=11.5, label=None):
    box(s, x, y, w, h, CODEBG)
    if label:
        text(s, x + 0.2, y + 0.12, w - 0.4, 0.3, [label], size=10.5, color=MUTED, space=0)
        y, h = y + 0.3, h - 0.3
    text(s, x + 0.2, y + 0.15, w - 0.4, h - 0.3, src.strip("\n").split("\n"), size=size, font=MONO,
         color=INK, space=0)


def stat(s, x, y, w, big, label, color=NAVY, big_size=40, dark=False):
    text(s, x, y, w, 0.8, [big], size=big_size, color=color, space=0)
    s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    text(s, x, y + 0.82, w, 0.7, [label], size=13, color="C9D1E0" if dark else MUTED, space=0)


def card(s, x, y, w, h, head, body, tint=BLUE_T, head_color=NAVY, size=14):
    box(s, x, y, w, h, tint)
    text(s, x + 0.25, y + 0.2, w - 0.5, h - 0.35,
         [[(head, {"bold": True, "color": head_color, "size": size + 2})]] + body, size=size, space=4)


def arrow(s, x, y, w=0.55, h=0.4, color=MUTED):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(color)
    a.line.fill.background()
    a.shadow.inherit = False


def chart(s, x, y, w, h, kind, cats, series, colors, title_txt=None, fmt='0.0', legend=True,
          vmin=None, vmax=None, font_size=12, reverse=False):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(font_size)
    ch.font.name = FONT
    ch.font.color.rgb = rgb(INK)
    ch.has_title = bool(title_txt)
    if title_txt:
        ch.chart_title.text_frame.text = title_txt
        tp = ch.chart_title.text_frame.paragraphs[0]
        tp.runs[0].font.size = Pt(font_size + 2)
        tp.runs[0].font.bold = True
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
    for ser, c in zip(ch.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(c)
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.overlap = -10
    pl.has_data_labels = True
    pl.data_labels.number_format = fmt
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E4E3DF")
    va.format.line.fill.background()
    va.tick_labels.font.color.rgb = rgb(MUTED)
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    ca = ch.category_axis
    ca.tick_labels.font.color.rgb = rgb(INK)
    ca.format.line.color.rgb = rgb("C8CCD2")
    if reverse:
        ca.reverse_order = True
    return ch


# ---------------------------------------------------------------- filminas -----
def s_portada(prs):
    s = new_slide(prs, dark=True)
    text(s, 0.8, 1.3, 11.5, 1.8, ["Reconstruir el diagrama de fases con 64 puntos"], size=44,
         color=WHITE, space=0)
    s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    text(s, 0.8, 2.75, 11, 0.6, ["Cómo funciona el loop autoexp, paso a paso"], size=22, color="C9D1E0")
    for k, (big, lab, c) in enumerate([
        ("64", "puntos consultados\n(0,39 % de 128×128)", ORANGE),
        ("> 38 dB", "el objetivo del 23/9", "7FB2F0"),
        ("41 dB", "logrado en los 9 g\n(media, con DIP)", "5FD3A6"),
        ("45 / 64", "puntos de KMC real\nya corridos (g = −4)", WHITE),
    ]):
        stat(s, 0.8 + k * 3.05, 4.4, 2.8, big, lab, color=c, big_size=44, dark=True)
    text(s, 0.8, 6.7, 11, 0.4, ["branch autoexp · worktree ~/Tesis-autoexp · código en autoexp/"],
         size=12, color="8A94A8")
    notes(s, """
Esta presentación es para entender el método, no para mostrar resultados (para eso está la 4).
La idea en una frase: cada punto del diagrama con KMC cuesta horas o días, así que en vez de
calcular la imagen entera se eligen muy pocos puntos (64) de forma inteligente y el resto se
reconstruye. El objetivo que fijamos el 23/9 fue: ≤ 64 puntos y más de 38 dB de PSNR.
Hoy: 40,6 dB de media en los 9 g sin DIP y 41,1 con DIP; y ya estamos corriendo KMC real para g=−4.
""")


def s_problema(prs):
    s = new_slide(prs)
    title(s, "El problema: cada píxel cuesta horas", step=0)
    image(s, os.path.join(FIG, "problema.png"), 0.6, 1.45, w=8.4, h=3.9)
    card(s, 9.3, 1.5, 3.45, 3.8, "Qué es la imagen", [
        "SoC_max en función de log ℓ (columnas) y log Ξ (filas).",
        "128 × 128 = 16 384 píxeles.",
        "La “verdad” para desarrollar: el modelo continuo (galpynostatic) en 9 g negativos.",
    ], size=13)
    for k, (big, lab, c) in enumerate([
        ("16 384", "píxeles de la imagen", NAVY),
        ("64", "consultas permitidas", ORANGE),
        ("min → días", "lo que tarda un punto de KMC (~1/ℓ)", NAVY),
    ]):
        stat(s, 0.6 + k * 4.2, 5.65, 3.9, big, lab, color=c, big_size=36)
    notes(s, """
Ejes: columnas = log ℓ de −4 a 2 (izquierda a derecha); filas = log Ξ de 2 (arriba) a −4 (abajo).
Dos propiedades físicas que se usan todo el tiempo:
1) El SoC NUNCA sube si bajás (menos Ξ) o si vas a la derecha (más corriente). Monotonía.
2) Debajo de cierto borde el SoC es exactamente 0 (el potencial corta enseguida): el "acantilado".
Los tres g muestran lo mismo: g=−4 tiene un frente seco, g=−0.5 una rampa más suave.
Por qué pocos puntos: una corrida de KMC tarda de minutos (ℓ alto) a días (log ℓ ≈ −3.5).
""")


def s_reglas(prs):
    s = new_slide(prs)
    title(s, "Las reglas del juego: nadie mira la imagen", step=1,
          sub="oracle.py y eval.py son archivos fijos: el loop de experimentación no los toca nunca")
    y = 2.0
    for k, (head, body, tint) in enumerate([
        ("Método", ["sampling.py + interp.py", "decide dónde medir y reconstruye"], BLUE_T),
        ("Oráculo", ["oracle.py", "responde el valor de un píxel, cuenta y anota"], ORANGE_T),
        ("Evaluador", ["eval.py", "chequea el registro contra la verdad, calcula PSNR/SSIM"], GREEN_T),
    ]):
        x = 0.6 + k * 4.25
        card(s, x, y, 3.5, 1.65, head, body, tint=tint, size=13)
        if k < 2:
            arrow(s, x + 3.6, y + 0.62)
    text(s, 4.0, 1.62, 1.6, 0.3, ["query(i, j)"], size=11, color=MUTED, font=MONO)
    text(s, 8.2, 1.62, 1.8, 0.3, ["queries.json"], size=11, color=MUTED, font=MONO)
    code(s, 0.6, 4.0, 7.3, 3.0, """
def query(self, points):
    new = {p for p in pts if p not in self.queried}
    if self.n_used + len(new) > self.budget:
        raise BudgetExceeded(...)
    for i, j in new:
        x = float(self.__img[i, j])   # imagen privada
        if self.noise > 0:            # opcional: ruido KMC
            x += self.noise * rng.standard_normal()
        self.queried[(i, j)] = x
""", size=11.5, label="autoexp/oracle.py")
    text(s, 8.3, 4.05, 4.4, 3.0, [
        [("Pseudo-puntos permitidos: ", {"bold": True}), ("valores inventados a partir de lo medido; "
                                                        "no gastan presupuesto.", {})],
        [("Anti-trampa: ", {"bold": True}), ("eval.py verifica que cada valor anotado coincida con la "
                                            "imagen real.", {})],
        [("Desarrollo ", {"bold": True}), ("en 3 g (−4, −2, −0.5); ", {}), ("validación ", {"bold": True}),
         ("en los 9. Siempre media + peor g.", {})],
    ], size=14, bullets=True, space=10)
    notes(s, """
Esta es la respuesta a "¿cómo sabés que no hiciste trampa?".
El método no tiene acceso a la imagen: solo puede llamar oracle.query((i, j)) y cada punto nuevo
gasta presupuesto. Todo queda en queries.json. Después eval.py, que es independiente, verifica:
que no haya más de 64 consultas, que no haya duplicadas, y que cada valor anotado sea el real
(así no se puede meter un valor inventado haciéndolo pasar por medido).
Métricas: PSNR y SSIM de la imagen entera, y PSNR en la frontera (10 % de píxeles con más gradiente),
que pidieron el 17/9 porque el PSNR global lo dominan las mesetas.
Los pseudo-puntos sí están permitidos (por ejemplo "debajo del borde vale 0"), pero no cuentan
como observaciones.
""")


def s_baseline(prs):
    s = new_slide(prs)
    title(s, "Punto de partida: grilla + interpolación de scipy", step=2)
    image(s, os.path.join(FIG, "baseline.png"), 0.6, 1.4, w=12.1, h=3.9)
    vals = [("vecino más cercano", "19.1"), ("lineal", "21.9"), ("cúbica", "22.9"), ("TPS (RBF)", "24.4")]
    for k, (lab, v) in enumerate(vals):
        x = 0.6 + k * 3.1
        box(s, x, 5.55, 2.85, 1.35, ORANGE_T if k == 3 else CODEBG)
        text(s, x + 0.25, 5.62, 2.4, 0.7, [v + " dB"], size=30, color=ORANGE if k == 3 else NAVY, space=0)
        s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        text(s, x + 0.25, 6.35, 2.4, 0.4, [lab], size=13, color=MUTED, space=0)
    text(s, 0.6, 7.0, 12, 0.3, ["media en los 3 g de desarrollo (la figura de arriba es solo g = −2)"],
         size=11, color=MUTED, space=0)
    notes(s, """
Lo primero: ¿qué da lo más simple? Grilla 8×8 (64 puntos) + interpolar. Promedio en los 3 g de desarrollo:
vecino más cercano 19.1, lineal 21.9, cúbica 22.9, TPS (thin-plate spline, RBFInterpolator) 24.4.
La TPS es la mejor y queda como base de todo lo que sigue.
interp2d (lo que pidió el profe) ya no existe en scipy 1.14; con su motor interno da 18.7-24.1: lo mismo.
Referencia de DIP previo: 20.6 dB con 48 puntos en g=−4.
Mirá el mapa de error: TODO el error está en el borde. La TPS no puede hacer un salto: ondula
("cuentas") y deja fantasmas a los dos lados. Eso marca el camino.
""")


def s_perfil(prs):
    s = new_slide(prs)
    title(s, "La observación clave: siempre el mismo perfil", step=3)
    image(s, os.path.join(FIG, "perfil.png"), 0.6, 1.4, w=7.4, h=5.4)
    card(s, 8.4, 1.6, 4.35, 1.5, "En todos los g", ["meseta → rampa suave → salto a 0 exacto"], tint=BLUE_T, size=16)
    card(s, 8.4, 3.3, 4.35, 1.5, "El error está en el salto", ["las mesetas se reconstruyen casi solas"],
         tint=ORANGE_T, head_color="B4471C", size=16)
    card(s, 8.4, 5.0, 4.35, 1.8, "Entonces…", ["gastar puntos en UBICAR el borde con precisión, "
                                                "no en cubrir parejo"], tint=GREEN_T, head_color="137A55", size=16)
    notes(s, """
Corte vertical en la columna 20 para tres g. Siempre es igual: arriba una meseta (SoC ≈ 1),
después una rampa suave (más ancha cuando g → 0) y al final un salto seco a exactamente 0.
Casi todo el error de la interpolación está en ese salto. Con presupuesto fijo, conviene gastar
la mayor parte de los puntos en saber dónde está el borde, y muy pocos en las mesetas.
Toda la receta sale de esta idea.
""")


def s_muestreo(prs, info):
    e = info["etapas"]
    s = new_slide(prs)
    title(s, "Dónde medir: tres etapas", step=4, sub="sampling.bisect — la misma receta que usa el planificador de KMC")
    image(s, os.path.join(FIG, "muestreo_etapas.png"), 0.6, 1.7, w=12.1, h=3.8)
    for k, (head, body, tint, hc) in enumerate([
        ("1 · Grilla (%d)" % e["grilla"], "6 columnas × 5 filas: idea global y en qué columnas hay salto", BLUE_T, NAVY),
        ("2 · Bisección (%d)" % e["bisección"], "búsqueda binaria del acantilado en cada columna, hasta 1 px",
         ORANGE_T, "B4471C"),
        ("3 · Relleno (%d)" % e["relleno"], "donde la validación cruzada dice que más se equivoca", GREEN_T, "137A55"),
    ]):
        card(s, 0.6 + k * 4.15, 5.65, 3.85, 1.4, head, [body], tint=tint, head_color=hc, size=13)
    notes(s, """
El caso que sigue en todas las figuras es g = −2 con 64 puntos, con la receta final.
Etapa 1: grilla gruesa de 30 puntos (6 columnas × 5 filas). Sirve para tener una idea global y,
sobre todo, para ver en qué columnas hay un par de puntos vecinos donde uno vale ~0 y el otro no:
ahí cruza el acantilado.
Etapa 2: en cada una de esas columnas, búsqueda binaria vertical (siguiente filmina). 25 puntos.
Etapa 3: los que sobran van donde el modelo más se equivoca según validación cruzada. 9 puntos.
Con KMC real esto son %d tandas: 1 de grilla, varias de bisección, 2-3 de relleno.
""" % info["rondas"])


def s_biseccion(prs, info):
    s = new_slide(prs)
    title(s, "Bisección: el borde por búsqueda binaria", step=5)
    image(s, os.path.join(FIG, "biseccion.png"), 0.5, 1.4, w=6.9, h=4.2)
    code(s, 7.6, 1.45, 5.15, 3.2, """
# corchete: un extremo ~0 y el otro no
if (v0 < eps) != (v1 < eps):
    br.append([x, y0, v0, y1, v1])

# cada ronda: el medio de TODOS los
# corchetes abiertos, a la vez
while any(b[3] - b[1] > tol for b in br):
    pts = [((b[1]+b[3])//2, b[0]) for b in br]
    vals = oracle.query(pts)
    # achicar el corchete hacia el medio
""", size=11, label="autoexp/sampling.py · bisect()")
    text(s, 7.6, 4.85, 5.15, 2.3, [
        [("eps = 0.004: ", {"bold": True}), ("el PNG es de 8 bits (1/255 ≈ 0.0039 es “cero”).", {})],
        [("~5 consultas por columna: ", {"bold": True}), ("log₂(26 px) ≈ 5.", {})],
        [("En lote: ", {"bold": True}), ("una ronda = una tanda de KMC en paralelo.", {})],
    ], size=14, bullets=True, space=8)
    text(s, 0.6, 5.85, 6.8, 1.2, [
        [("En g = −4 ", {"bold": True}), ("(el caso “imposible” para DIP con pocos puntos) esto solo ya "
                                          "llevó de 24 a 34 dB.", {})]], size=15, color=NAVY)
    notes(s, """
Columna %d del caso g = −2. Los puntos azules son la grilla: el penúltimo vale ~0.9 y el último 0,
así que el borde está entre las filas 89 y 115 (corchete de 26 px).
Ronda 1: pregunto el medio (fila 102): vale 0, así que el borde está arriba. Ronda 2: medio del nuevo
corchete, y así. Con %d consultas el corchete queda en 1 px.
Detalles para defender: "cero" es v < 0.004 porque la imagen es PNG de 8 bits. Solo se biseca si el
lado no nulo es alto (≥ 0.05): arriba a la derecha el mapa se desvanece suave hasta ~0.02 y eso no es
un acantilado.
Las rondas son EN LOTE: todas las columnas piden su punto medio a la vez. Con KMC real, cada ronda es
una tanda de corridas en paralelo en el cluster, no una corrida por vez.
""" % (info["col_biseccion"], info["n_biseccion_col"]))


def s_loo(prs):
    s = new_slide(prs)
    title(s, "Relleno: medir donde el modelo más se equivoca", step=6)
    image(s, os.path.join(FIG, "loo.png"), 0.5, 1.35, w=6.2, h=5.7)
    text(s, 7.0, 1.5, 5.8, 1.9, [
        "Para cada punto medido arriba del borde:",
        [("1. ", {"bold": True}), ("sacarlo y reconstruir sin él,", {})],
        [("2. ", {"bold": True}), ("medir cuánto se equivoca en ese lugar (error leave-one-out),", {})],
        [("3. ", {"bold": True}), ("interpolar esos errores a toda la imagen.", {})],
    ], size=15, space=5)
    code(s, 7.0, 3.55, 5.75, 1.75, """
k = argmax( error_LOO ** power
            * distancia_al_punto_mas_cercano )
""", size=12, label="autoexp/sampling.py · loo_fill()")
    text(s, 7.0, 5.5, 5.75, 1.5, [
        "Error alto y lejos de lo medido → ahí va el punto. De a 4 por ronda.",
        [("Se excluyen 4 px sobre el borde ", {"bold": True}),
         ("(ya los resolvió la bisección). Sin eso, g = −4 cae de 45.7 a 37.4 dB.", {})],
    ], size=14, bullets=True, space=8)
    notes(s, """
Mapa de calor = error de validación cruzada con los 55 puntos que había al terminar la bisección.
Leave-one-out: saco un punto, reconstruyo con los demás y miro cuánto le erro a ese punto. Donde
el modelo le erra a sus propios puntos, es donde menos información tiene.
El criterio es error × distancia: así no se amontonan los puntos todos en el mismo lugar.
Las estrellas son los 4 que eligió en la primera ronda (a la derecha arriba, en la zona que
se desvanece, que cae entre columnas de la grilla).
Antes se probó un relleno por gradiente: el LOO le ganó (38.65 dB de media en 9 g cuando se agregó).
""")


def s_borde(prs):
    s = new_slide(prs)
    title(s, "Reconstruir (1): ubicar el borde", step=7, sub="interp.cliff — el “modelo de acantilado”")
    image(s, os.path.join(FIG, "borde.png"), 0.5, 1.7, w=5.9, h=5.4)
    for k, (head, body) in enumerate([
        ("Triangulación de Delaunay", "une cada punto con sus vecinos (líneas blancas)."),
        ("Aristas que cruzan", "cortas (≤ 4 px), con un extremo = 0 y el otro > 0 → su punto medio es un punto del borde (×)."),
        ("RANSAC", "prueba todos los subconjuntos mínimos y gana el que más puntos deja a ≤ 3 px: un cruce mal detectado no tuerce la recta."),
        ("Grado automático", "recta, parábola o cúbica: el más bajo que logra el máximo de inliers; si falla, curva por tramos (PCHIP)."),
    ]):
        y = 1.75 + k * 1.33
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(6.8), Inches(y + 0.05), Inches(0.42), Inches(0.42))
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(BLUE)
        c.line.fill.background()
        c.shadow.inherit = False
        c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        _run(c.text_frame.paragraphs[0], str(k + 1), 13, True, WHITE)
        text(s, 7.45, y, 5.3, 1.25, [[(head, {"bold": True, "size": 16, "color": NAVY})], body], size=13.5,
             space=2)
    notes(s, """
Con los 64 puntos ya medidos hay que armar la imagen. Primero, el borde.
Se triangulan los puntos (Delaunay). Si una arista es corta y une un punto que vale 0 con uno que no,
el borde pasa por el medio: esos son los puntos de borde (cruces naranjas).
Por esos puntos se ajusta fila = f(columna). Con RANSAC en vez de mínimos cuadrados comunes: se prueban
todos los subconjuntos mínimos y gana el que tiene más puntos cerca; así un cruce malo no mueve la recta.
El grado sale solo (el más bajo que alcanza). En los 9 g del continuo sale siempre una recta; en los
diagramas deformados (arcos, S) sale curva. Eso permite usarlo cuando no sabemos la forma del KMC.
""")


def s_recon(prs, info):
    p = info["pasos"]
    s = new_slide(prs)
    title(s, "Reconstruir (2): cada idea suma", step=8, sub="mismos 64 puntos en las cuatro columnas (g = −2)")
    image(s, os.path.join(FIG, "recon_pasos.png"), 0.6, 1.6, w=8.1, h=5.6)
    items = [
        ("Cero debajo del borde", "y TPS solo con los puntos de arriba: sin salto que la TPS no pueda hacer.",
         p[1][1] - p[0][1]),
        ("y comprimida + rampa alineada", "la rampa corre paralela al borde: se interpola en coordenadas "
                                          "“a lo largo / a través” del borde.", p[2][1] - p[1][1]),
        ("Monotonía", "cotas exactas + proyección a mapas no crecientes (siguiente).", p[3][1] - p[2][1]),
    ]
    for k, (head, body, d) in enumerate(items):
        y = 1.65 + k * 1.85
        box(s, 9.0, y, 3.75, 1.65, [ORANGE_T, BLUE_T, GREEN_T][k])
        text(s, 9.2, y + 0.12, 1.4, 0.5, ["+%.1f dB" % d], size=20, color=[ORANGE, BLUE, "137A55"][k], space=0)
        s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        text(s, 9.2, y + 0.58, 3.4, 1.05, [[(head, {"bold": True})], body], size=12, space=2)
    notes(s, """
Misma entrada (64 puntos), cuatro reconstrucciones. Fila de abajo = error.
TPS común: %.1f dB, ondula sobre el borde.
+ cero debajo del borde: debajo vale 0 exacto; arriba, TPS usando SOLO los puntos de arriba. %.1f dB.
+ compresión vertical y rampa alineada: lejos del borde la estructura depende casi solo de x, así que
la distancia vertical se multiplica por 0.6. Cerca del borde hay una rampa paralela al borde, pero
solo la medimos en 6 columnas: se interpola en coordenadas rotadas (a lo largo del borde comprimido)
para que el perfil medido en una columna se traslade a las vecinas. El ancho de esa franja es adaptativo:
si justo arriba del borde el valor es alto (frente seco, g=−4) la franja es angosta. %.1f dB.
+ monotonía: %.1f dB. En g = −2 es el peor de los 3 g de desarrollo; en g = −4 la misma receta da ~45 dB.
""" % (p[0][1], p[1][1], p[2][1], p[3][1]))


def s_monotonia(prs, info):
    s = new_slide(prs)
    title(s, "Reconstruir (3): la física ya dice mucho", step=9, sub="el SoC no sube hacia abajo ni hacia la derecha")
    image(s, os.path.join(FIG, "monotonia.png"), 0.5, 1.6, w=5.9, h=5.2)
    stat(s, 6.8, 1.7, 5.9, "%d %%" % round(100 * info["frac_mono"]),
         "del mapa ya tiene su intervalo [L, U] de ancho ≤ 0.05 con solo 64 puntos, antes de interpolar", color=GREEN)
    code(s, 6.8, 3.35, 5.95, 1.9, """
# p medido abajo-derecha de q  ->  f(q) >= v(p)
L = max.accumulate(max.accumulate(M[::-1,::-1]))
U = min.accumulate(min.accumulate(N))
rec = clip(rec, L, U)
rec = monotone_2d(rec)   # Dykstra + PAVA
""", size=11.5, label="autoexp/interp.py · monotone_bounds / monotone_2d")
    text(s, 6.8, 5.45, 5.95, 1.6, [
        [("Cotas: ", {"bold": True}), ("cada punto medido acota toda una esquina del mapa.", {})],
        [("Proyección: ", {"bold": True}), ("se alterna regresión isotónica por columnas y por filas hasta "
                                            "que el mapa es monótono; borra las ondulaciones.", {})],
    ], size=13.5, bullets=True, space=8)
    notes(s, """
Como el SoC no sube ni bajando ni yendo a la derecha, un punto medido q que está abajo-derecha de p
dice f(p) >= v(q); uno arriba-izquierda dice f(p) <= v(q). Eso da, para cada píxel, un intervalo
[L, U] EXACTO (no es una estimación). El mapa muestra el ancho U − L: negro = ya se sabe.
Con 64 puntos, en el %d %% del mapa el intervalo mide 0.05 o menos.
Se usa de dos formas: se recorta la interpolación a [L, U] y después se proyecta al conjunto de mapas
monótonos (algoritmo de Dykstra, que alterna regresión isotónica PAVA por columnas y por filas).
Si te preguntan "¿no estás metiendo conocimiento a mano?": sí, conocimiento FÍSICO (monotonía, SoC=0
debajo del corte), no de la imagen particular. Y la próxima filmina muestra que se chequea con los datos.
""" % round(100 * info["frac_mono"]))


def s_auto(prs):
    s = new_slide(prs)
    title(s, "¿Y si el KMC tiene otra forma?", step=10,
          sub="interp.auto: el método chequea cada supuesto con los propios puntos; si no se cumple, lo apaga")
    for k, (q, yes, no) in enumerate([
        ("¿Algún par de puntos viola la monotonía?", "no → cotas + proyección", "sí → sin monotonía"),
        ("¿≥ 90 % de lo de abajo del borde vale ~0?", "sí → modelo de acantilado", "no → TPS genérica"),
        ("¿Cuánto comprimir en vertical?", "4 valores, gana el de menor", "error de validación cruzada"),
    ]):
        y = 1.95 + k * 1.2
        box(s, 0.6, y, 4.3, 1.0, BLUE_T)
        text(s, 0.8, y, 3.95, 1.0, [q], size=14, color=NAVY, anchor=MSO_ANCHOR.MIDDLE, space=0)
        arrow(s, 5.0, y + 0.3, w=0.45, h=0.38)
        text(s, 5.6, y + 0.12, 2.3, 0.8, [[(yes, {"bold": True, "color": "137A55"})], no], size=12.5, space=2)
    chart(s, 8.1, 1.8, 4.7, 5.3, XL_CHART_TYPE.COLUMN_CLUSTERED,
          ["Fuera de muestra\n(g=+2, +4, Gauss,\nRosenbrock, Himmelblau)", "Diagramas\ndeformados (15)"],
          [("grilla + TPS", (33.8, 24.5)), ("receta que se chequea", (36.0, 33.4))],
          [MUTED, ORANGE], title_txt="PSNR medio [dB], mapas nunca usados", vmin=0, vmax=40, font_size=11)
    text(s, 0.6, 5.75, 7.2, 1.3, [
        [("Por qué importa: ", {"bold": True}), ("todo se afinó con el modelo continuo. En mapas que el método "
                                                 "nunca vio, sigue ganando: no está sobreajustado a esos 9.", {})]],
        size=14.5, color=INK)
    notes(s, """
La objeción natural: "todo esto está hecho a medida de los mapas del continuo; el KMC puede ser distinto".
Respuesta: interp.auto no asume nada a ciegas. Con los mismos puntos medidos chequea:
- si algún par de puntos viola la monotonía (con tolerancia 0.02 + 3σ): si sí, se apaga la monotonía;
- si de verdad hay acantilado (al menos 3 puntos debajo del borde y el 90 % vale ~0): si no, TPS común;
- la compresión vertical la elige por validación cruzada entre 1.0, 0.8, 0.6, 0.4.
Prueba fuera de muestra (mapas nunca usados para afinar): 36.0 vs 33.8 del baseline.
Diagramas deformados (el acantilado corrido, inclinado, en arco, en S): 33.4 vs 24.5.
En los 9 g originales da idéntico (sale recta en todos).
""")


def s_fusion(prs):
    s = new_slide(prs)
    title(s, "Dónde entra DIP: fusión con la TPS", step=11, sub="autoexp/fuse.py")
    image(s, os.path.join(FIG4, "zonas_tps_dip.png"), 0.6, 1.55, w=12.1, h=3.55)
    code(s, 0.6, 5.3, 4.6, 1.75, """
# d = distancia (px) sobre el borde
w = exp(-(d / B) ** 2)
final = monotone_2d(w * TPS + (1 - w) * DIP)
""", size=11.5, label="cerca del borde manda la TPS; lejos, DIP")
    card(s, 5.45, 5.3, 3.55, 1.75, "Guarda", ["si DIP no reproduce los puntos reales (RMS > 0.02), DIP falló "
                                               "→ solo TPS"], tint=ORANGE_T, head_color="B4471C", size=12.5)
    card(s, 9.2, 5.3, 3.55, 1.75, "Mejor de 2", ["dos semillas de DIP; queda la de menor residuo en los "
                                                  "puntos medidos"], tint=GREEN_T, head_color="137A55", size=12.5)
    notes(s, """
Cada uno es bueno en una zona distinta. La TPS con rampa alineada es muy precisa pegada al borde
(DIP ahí deja escalones o huecos). DIP es más lisa lejos del borde (mesetas y zona que se desvanece),
donde la TPS ondula. Se mezclan con un peso que depende de la distancia al borde (B = 20 px).
A DIP se le pasan los puntos reales + pseudo-ceros debajo del borde (seguro valen 0, no gastan presupuesto).
Dos protecciones, ninguna mira la imagen real:
- guarda: si DIP no ajusta los puntos REALES de arriba del borde (RMS > 0.02) se descarta. En los 9 g,
  cuando DIP anda el residuo es ≤ 0.008 y cuando falla ≥ 0.03: el corte es limpio.
- mejor de 2: DIP varía entre semillas; se corren 2 y se elige por residuo en los puntos.
DIP corre en Mendieta (GPU A30), nunca local.
""")


def s_recorrido(prs):
    s = new_slide(prs)
    title(s, "El recorrido completo, con el mismo presupuesto", sub="PSNR en los 3 g de desarrollo (autoexp/runs/*/score.json)")
    labels = [l.replace("\n", " ") for l, _ in STEPS]
    st = [dev_stats(r) for _, r in STEPS]
    ch = chart(s, 0.6, 1.6, 12.1, 5.6, XL_CHART_TYPE.BAR_CLUSTERED, labels,
               [("PSNR medio (3 g)", [round(a, 1) for a, _ in st]), ("peor g", [round(b, 1) for _, b in st])],
               [BLUE, ORANGE], vmin=15, vmax=46, font_size=11, reverse=True)
    ch.plots[0].gap_width = 40
    notes(s, """
Cada barra es una corrida real del loop, todas con 64 puntos y el mismo evaluador.
Resumen de lo que más sumó: bisección del frente (+2.5), acantilado a cero (+3.3), relleno por
validación cruzada (+3.0), monotonía, RANSAC y la fusión con DIP al final.
El objetivo de 38 dB en el PEOR g se pasa con la fusión (39.5).
Regla del loop: un cambio por vez, se mide, y si empeora se revierte con git. Está todo anotado en
autoexp/experimentos.md (la bitácora) y leaderboard.csv.
""")


def s_resultados(prs):
    s = new_slide(prs)
    title(s, "Resultados: lo que hay que saber de memoria", sub="validación en los 9 g negativos")
    for k, (big, lab, c) in enumerate([
        ("40.6 dB", "64 puntos, sin DIP\n(peor g 36.2)", NAVY),
        ("41.1 dB", "64 puntos, fusión con DIP\n(peor g 37.3; 8/9 g ≥ 38)", ORANGE),
        ("84 / 88", "puntos para ≥ 40 dB en los 9 g\ncon DIP / sin DIP", GREEN),
        ("× 4", "menos puntos que DIP previo\n(~328, y ese miraba el mapa)", BLUE),
    ]):
        stat(s, 0.6 + k * 3.15, 1.7, 2.95, big, lab, color=c, big_size=38)
    chart(s, 0.6, 3.55, 6.3, 3.6, XL_CHART_TYPE.COLUMN_CLUSTERED, ["σ = 0.01", "σ = 0.03", "σ = 0.05"],
          [("TPS sola", (38.3, 34.2, 31.4)), ("fusión TPS + DIP", (39.0, 34.9, 32.6))], [MUTED, ORANGE],
          title_txt="Con ruido tipo KMC (4 g) [dB]", vmin=25, vmax=42, font_size=11)
    card(s, 7.3, 3.6, 5.45, 3.5, "Cómo leerlo", [
        "Con más ruido, DIP suma más (sobre todo en SSIM): es el argumento para mantenerlo.",
        "Banda honesta sin DIP, moviendo la grilla: 38.9 ± 1.3 dB.",
        "Lo que falta está en la franja de 0-12 px sobre el borde, y ahí TPS y DIP fallan igual: "
        "falta información, no un mejor reconstructor.",
    ], size=13)
    notes(s, """
Números clave:
- 64 puntos, 9 g: 40.6 dB sin DIP (peor 36.2); 41.1 con fusión + guarda (peor 37.3), 8 de 9 g ≥ 38
  (solo g = −1.5 queda abajo).
- Para ≥ 40 dB en los 9 g hacen falta 84 puntos con DIP (43.4 / 40.2) u 88 sin DIP (42.9 / 41.2).
  Es ~0.5 % de la imagen. El DIP previo con máscara frontier_mix necesitaba ~328, y esa máscara usaba
  el mapa denso para elegir puntos.
- Ruido: el oráculo puede sumar ruido gaussiano de desvío σ; el puntaje se mide contra el mapa limpio.
  DIP suma más cuanto más ruido. En KMC medimos σ ≈ 0.02 entre réplicas para g = −4.
- Robustez: si corro la grilla (otro offset), sin DIP da 38.9 ± 1.3. El 40.6 es con el offset por defecto.
Descartados (por si preguntan): proceso gaussiano/kriging, sondas de rampa, Optuna sobre DIP solo
(mejor ~29.9 con grilla de 64).
""")


def s_kmc_loop(prs):
    s = new_slide(prs)
    title(s, "Con KMC real: el mismo muestreo, por tandas", step=12, sub="autoexp/kmc_planner.py + kmc_auto.py")
    steps = [("next", "qué puntos\ncorrer", BLUE_T), ("cluster", "Serafín /\nMulatona", ORANGE_T),
             ("kmc_results", "leer SoC\n+ réplicas", GREEN_T), ("add", "cargar al\nestado", BLUE_T)]
    for k, (h, b, t) in enumerate(steps):
        x = 0.6 + k * 3.1
        box(s, x, 1.75, 2.5, 1.35, t)
        text(s, x + 0.2, 1.82, 2.1, 1.25, [[(h, {"bold": True, "font": MONO, "size": 15, "color": NAVY})],
                                            b], size=13, align=PP_ALIGN.CENTER, space=2)
        if k < 3:
            arrow(s, x + 2.55, 2.25, w=0.5, h=0.38)
    text(s, 0.6, 3.2, 12, 0.4, ["Se repite hasta 64 puntos; al final: reconstrucción TPS + DIP (Mendieta) y 10 puntos de control."],
         size=13, color=MUTED)
    code(s, 0.6, 3.8, 6.4, 3.25, """
class ReplayOracle:   # misma interfaz que Oracle
    def query(self, points):
        missing = [p for p in new
                   if p not in self.known]
        if missing:
            raise NeedPoints(missing)  # = la tanda
        ...

# next: se corre bisect() desde cero
# con lo ya conocido; donde frena, esa
# es la tanda siguiente (es determinista)
""", size=11.5, label="autoexp/kmc_planner.py")
    text(s, 7.3, 3.85, 5.45, 3.2, [
        [("rc = 0: ", {"bold": True}), ("terminó (sin filas = SoC 0).", {})],
        [("rc = 124: ", {"bold": True}), ("cortado por tiempo → el SoC real es ≥ el último: cota.", {})],
        [("Pseudo por monotonía: ", {"bold": True}),
         ("un punto que no termina vale al menos lo medido a su derecha.", {})],
        [("Carrera: ", {"bold": True}), ("cada tanda va a Mulatona y a Serafín; gana la que termina "
                                         "(Mulatona desaloja jobs).", {})],
    ], size=13.5, bullets=True, space=8)
    notes(s, """
El planificador reutiliza EXACTAMENTE el mismo muestreo. El truco es el ReplayOracle: tiene la misma
interfaz que el oráculo, pero responde con los resultados de KMC ya cargados. En cada "next" se corre
el muestreo desde cero; cuando pide un punto que todavía no se corrió, tira NeedPoints y esos puntos
son la tanda siguiente. Como el muestreo es determinista, siempre llega al mismo lugar.
kmc_results lee datos-*.dat y un estado.txt con el código de salida que escribe el slurm. Se usa el
código de salida y no la última E porque el último intervalo a veces no se imprime.
Los puntos de log ℓ ≈ −3.5 tardan días: no terminan. Se cargan como pseudo: el SoC no sube hacia la
derecha, así que vale al menos lo que midió el vecino de la derecha en la misma fila (si ese vecino dio 0,
se usa la altura del escalón más cercana, 0.765).
kmc_auto corre local en segundo plano, se relanza solo con crontab si se reinicia la PC.
""")


def s_kmc_estado(prs):
    s = new_slide(prs)
    title(s, "g = −4 con KMC real: terminado", step=13)
    image(s, os.path.join(REPO, "autoexp", "kmc", "g-4", "avance.png"), 0.5, 1.3, w=6.4, h=5.9)
    for k, (big, lab, c) in enumerate([
        ("64 / 64", "puntos en 9 tandas\n(56 medidos + 8 pseudo)", NAVY),
        ("σ ≈ 0.02", "ruido entre réplicas de KMC", ORANGE),
        ("0.027", "RMS en 10 puntos de control\ncon TPS + DIP (máx 0.068)", GREEN),
        ("0.043", "RMS con TPS sola;\nel continuo como sustituto: 0.14", BLUE),
    ]):
        stat(s, 7.3 + (k % 2) * 2.85, 1.45 + (k // 2) * 1.85, 2.7, big, lab, color=c, big_size=34)
    card(s, 7.3, 5.25, 5.45, 1.85, "Para consultar", [
        "Con log Ξ ≥ 1.4 y log ℓ ≤ −3.5 el KMC da SoC ≈ 0 en ~3 min (el continuo da ~1). "
        "Reproducible: 6 de 6 semillas."], tint=ORANGE_T, head_color="B4471C", size=13)
    notes(s, """
Cada número en el mapa es la tanda en la que se corrió ese punto. Tanda 1 = grilla de 30 (Serafín);
tandas 2 a 6 = bisección (~5 puntos cada una) en las columnas donde cruza el borde; 7 a 9 = relleno.
Los punteados son pseudo (cortados por tiempo, cargados por monotonía). El recuadro es el job largo
(log ℓ = −3.53), todavía pendiente: cuando termine reemplaza a los pseudo y se rehace el final.
Control: 10 puntos que NO se usaron para reconstruir, corridos con KMC después. Error RMS del mapa
final contra esos puntos: 0.027 (fusión TPS + DIP), 0.043 (TPS con restricciones sola), 0.14 si se
usara el mapa del continuo con g = −4 como sustituto. Es decir: medir 64 puntos y reconstruir es 5
veces mejor que suponer que el KMC es igual al continuo.
""")


def fade(s):
    """Transicion de fundido (secuencias paso a paso)."""
    from pptx.oxml import parse_xml
    s._element.append(parse_xml('<p:transition xmlns:p="http://schemas.openxmlformats.org/'
                                'presentationml/2006/main" spd="med"><p:fade/></p:transition>'))


def _secuencia(titulo, step, sub, frames, caps, nota):
    def build(prs):
        for k, (img, cap) in enumerate(zip(frames, caps), 1):
            s = new_slide(prs)
            title(s, titulo, step=step, sub=sub)
            image(s, img, 0.5, 1.7, w=12.3, h=4.95)
            text(s, 0.6, 6.7, 12.1, 0.6, [cap], size=14, color=INK, space=0)
            notes(s, "Cuadro %d de %d. %s\n\n%s" % (k, len(frames), cap, nota))
            fade(s)
    return build


def s_kmc_tandas(prs, info):
    n = len(figuras_kmc.CAP_TANDAS)
    _secuencia("Cómo se va armando el mapa, tanda a tanda", 14,
               "g = −4, KMC real · círculos naranjas = puntos nuevos · línea punteada = borde detectado",
               [os.path.join(FIG, "kmc_tandas_g-4_%d.png" % k) for k in range(1, n + 1)], figuras_kmc.CAP_TANDAS, """
Cada panel es el mapa que se podía armar en ese momento con la misma receta (TPS con restricciones
físicas). El borde recién aparece en la tanda 4: con σ = 0.02 hace falta un par 0 / alto a ≤ 4 px y
que cerca del lado alto algo valga ≥ 0.1. Lo que más cambia el mapa son las tandas de bisección;
el relleno retoca detalles en la zona que se desvanece.""")(prs)


def s_kmc_pasos(prs, info):
    _secuencia("De los 64 puntos al mapa final: los 8 pasos", 15, "g = −4, KMC real · autoexp/kmc_dip.py",
               [os.path.join(FIG, "kmc_pasos_g-4_%d.png" % k) for k in range(1, 9)], figuras_kmc.CAP_PASOS, """
Orden importante: primero se ubica el borde con los puntos y DESPUÉS se hace la TPS solo con los puntos
de arriba. No es "TPS y después recortar": si la TPS usara también los ceros, chorrearía hacia la meseta
(panel 2). Recta del borde: fila = %.2f·col + %.1f. DIP no cierra todo: se mezcla por zonas, y si no
reproduce los puntos reales (RMS > 0.02 + 2σ) se descarta.""" % tuple(info["kmc_pasos"]["coef"]))(prs)


def s_scipy_metodos(prs, info):
    m = info["scipy_metodos"]
    s = new_slide(prs)
    title(s, "Las cuatro interpolaciones de scipy, lado a lado",
          sub="misma grilla 8×8 (64 puntos), g = −2 · arriba la reconstrucción, abajo el |error|")
    image(s, os.path.join(FIG, "scipy_metodos.png"), 0.4, 1.7, w=12.5, h=4.95)
    text(s, 0.6, 6.7, 12.1, 0.6, ["Todas fallan en el mismo lugar: el borde. Vecino más cercano hace escalones, "
                                  "lineal y cúbica dientes de sierra; la TPS es la más lisa y la que menos error deja."],
         size=14, color=INK, space=0)
    notes(s, """
Lo que pidió el profe en el punto 6 de la reunión del 17/9: interpolar con scipy y comparar.
Mismos 64 puntos en grilla 8×8, cuatro métodos (autoexp/interp.py, función reconstruct):
- vecino más cercano (griddata nearest): %.1f dB. Cada píxel copia el punto más cercano: bloques.
- lineal (griddata linear, triangula los puntos): %.1f dB. Dientes de sierra sobre el borde.
- cúbica (griddata cubic, Clough-Tocher): %.1f dB. Más suave, pero los mismos dientes.
- TPS (RBFInterpolator, thin_plate_spline): %.1f dB. La más lisa; ondula un poco y redondea el frente.
(Estos dB son de g = −2; la media en los 3 g de desarrollo está en la filmina anterior: 19.1 / 21.9 / 22.9 / 24.4.)
Mirá la fila de abajo: en los cuatro el error está casi todo en el borde inclinado. Ninguna interpolación
puede hacer un salto seco con puntos a 16 píxeles de distancia. De ahí salen las dos ideas que siguen:
medir más cerca del borde (bisección) y meter el salto a mano (cero debajo del borde).
""" % tuple(p for _, p in m))


def s_tps(prs):
    s = new_slide(prs)
    title(s, "Qué es la TPS (thin-plate spline)", sub="la mejor de las interpolaciones de scipy: la base de todo lo que sigue")
    image(s, os.path.join(FIG, "tps_que_es.png"), 0.5, 1.7, w=7.6, h=2.9)
    text(s, 0.6, 4.75, 7.4, 1.0, [
        [("Una chapa fina clavada a la altura de cada punto: ", {"bold": True}),
         ("toma la forma que menos se curva. Minimiza ∬ (f_xx² + 2f_xy² + f_yy²).", {})]], size=14)
    code(s, 0.6, 5.65, 7.4, 1.5, """
f(x) = a0 + a1·x + a2·y + Σ w_i · r_i² log r_i
RBFInterpolator(ij / s, v, kernel="thin_plate_spline",
                smoothing=1e-4)(q / s)
""", size=11.5, label="scipy.interpolate · autoexp/interp.py:35")
    card(s, 8.4, 1.75, 4.35, 2.55, "Lo bueno", [
        "Pasa por los puntos (o casi, con suavizado para el ruido).",
        "Sin parámetros que afinar; 64 puntos se resuelven en milisegundos."], tint=GREEN_T, head_color="137A55",
         size=13)
    card(s, 8.4, 4.5, 4.35, 2.65, "Ojo con el nombre", [
        "Sola, en un salto se pasa de largo: 24 dB.",
        "De acá en adelante, “TPS” = TPS con restricciones físicas (cero debajo del borde + monotonía): "
        "~40 dB. Lo que la hace andar es la física, no el interpolador."], tint=ORANGE_T, head_color="B4471C",
         size=13)
    notes(s, """
TPS = thin-plate spline (spline de placa delgada). Método clásico: Duchon (1976), Bookstein (1989).
Imaginá una chapa metálica finita clavada a la altura medida en cada punto: toma la forma de mínima
energía de flexión. Matemáticamente, la función que pasa por los puntos y minimiza la integral de las
segundas derivadas al cuadrado. La solución es un plano más un "bulto" r² log r centrado en cada punto;
los pesos salen de un sistema lineal de N×N (N = 64).
Con smoothing > 0 deja de pasar exacto por los puntos: sirve para el ruido del KMC.
Se usa de scipy: scipy.interpolate.RBFInterpolator con kernel="thin_plate_spline" (autoexp/interp.py:35).
Llegó por el punto 6 de la reunión del 17/9 (interpolación de scipy vs DIP): fue la mejor de todas.
IMPORTANTE para la reunión: la TPS sola NO le gana a DIP (24.4 vs ~29.9 de DIP en grilla). Lo que le
gana es la TPS + muestreo por tandas + cero debajo del borde + monotonía (~39-40 dB). Llamarla
"TPS con restricciones físicas" para que no haya confusión.
""")


def s_continuo(prs, info):
    c = info["continuo"]
    s = new_slide(prs)
    title(s, "La receta en los 11 g del continuo, con ruido tipo KMC",
          sub="64 puntos del planificador de KMC, σ = 0.005; el chequeo de supuestos decide si hay borde")
    image(s, os.path.join(FIG, "tps_continuo_g.png"), 0.5, 1.7, w=12.3, h=4.1)
    neg = [r for r in c if r[0] <= 0]
    pos = [r for r in c if r[0] > 0]
    card(s, 0.6, 5.95, 6.0, 1.3, "g ≤ 0: hay borde", [
        "TPS sola %.1f dB → con restricciones %.1f dB de media."
        % (np.mean([r[1] for r in neg]), np.mean([r[2] for r in neg]))], size=13)
    card(s, 6.75, 5.95, 6.0, 1.3, "g > 0: no hay borde", [PRESUPUESTO_TXT], tint=ORANGE_T, head_color="B4471C",
         size=13)
    notes(s, """
Cada columna es un g; fila 1 la verdad, fila 2 la TPS sola, fila 3 la TPS con restricciones físicas
(interp.auto: chequea con los puntos si hay borde y si se cumple la monotonía).
%s
Cuanto más suave el mapa, mejor anda la TPS sola (no hay salto que la haga ondular).
En g = +2 y +4 los puntos no muestran ningún par 0 / alto cercano: el corte no se activa solo
y queda TPS + monotonía. No es que el corte sea contraproducente: simplemente no se usa.
%s
Ojo: estos números son con ruido σ = 0.005 y una sola tirada; sin ruido, en los 9 g negativos la
misma receta da 40.6 dB de media (VAL9_ransac_s1e-4). El ruido cuesta ~1.3 dB, porque ubica peor el borde.
""" % ("\n".join("g = %+.1f: TPS sola %.1f dB, con restricciones %.1f dB%s" % (g, a, b, "" if ok else " (sin borde)")
                 for g, a, b, ok in c), PRESUPUESTO_NOTA))


def s_preguntas(prs):
    s = new_slide(prs)
    title(s, "Preguntas probables")
    qa = [
        ("¿Cómo sabés que no hiciste trampa?", "Oráculo que cuenta y anota; evaluador independiente que lo "
                                                "verifica contra la verdad."),
        ("¿Vale para KMC si se afinó con el continuo?", "Parcialmente: por eso la prueba fuera de muestra, los "
                                                          "deformados, el ruido y los 10 puntos de control."),
        ("¿Por qué no calibrar g y listo?", "Branch physics-calibration: 34 dB limpio, pero supone KMC = "
                                             "continuo con otro g, y el borde parece corrido."),
        ("¿Para qué DIP si la TPS llega?", "Sin ruido +0.5-1 dB; con ruido suma más, y con 84 puntos es lo "
                                            "que hace pasar los 9 g sobre 40."),
        ("¿Por qué no aleatorio o Halton?", "Probado: Halton 23.1, peor que la grilla. El muestreo activo "
                                             "gana porque la información no está repartida pareja."),
        ("¿Qué limita hoy?", "La franja de 0-12 px sobre el borde: falta información, no un mejor "
                              "reconstructor."),
    ]
    for k, (q, a) in enumerate(qa):
        x = 0.6 + (k % 3) * 4.12
        y = 1.55 + (k // 3) * 2.8
        card(s, x, y, 3.85, 2.55, q, [a], tint=[BLUE_T, ORANGE_T, GREEN_T][k % 3],
             head_color=[NAVY, "B4471C", "137A55"][k % 3], size=15)
    notes(s, """
Respuestas cortas para tener a mano. La versión larga de cada una está en las notas de las filminas
anteriores (reglas: 1; fuera de muestra: 10; DIP: 11; resultados: la de resultados).
""")


def s_cierre(prs):
    s = new_slide(prs, dark=True)
    text(s, 0.8, 1.2, 11.7, 0.8, ["En una frase"], size=22, color="C9D1E0")
    text(s, 0.8, 1.9, 11.7, 2.4, ["Medir donde está la información (el borde), reconstruir con lo que "
                                  "dice la física (cero abajo, monotonía) y dejar que DIP alise el resto."],
         size=32, color=WHITE)
    s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    for k, (h, b) in enumerate([
        ("grilla → bisección → relleno LOO", "dónde medir"),
        ("borde RANSAC + TPS + monotonía", "cómo reconstruir"),
        ("fusión con DIP + guarda", "dónde suma la red"),
    ]):
        x = 0.8 + k * 4.0
        text(s, x, 4.9, 3.7, 1.4, [[(b, {"color": "8A94A8", "size": 13})],
                                   [(h, {"bold": True, "color": [ORANGE, "7FB2F0", "5FD3A6"][k], "size": 17})]],
             space=4)
    notes(s, """
Tres piezas: dónde medir (grilla, bisección, relleno por validación cruzada), cómo reconstruir
(borde por RANSAC, TPS solo arriba, cero abajo, monotonía) y dónde entra DIP (fusión lejos del borde,
con guarda). Todo el código está en autoexp/, la bitácora en autoexp/experimentos.md.
""")


def main():
    info = figuras.main()
    info.update(figuras_kmc.main())
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    s_portada(prs)
    s_problema(prs)
    s_reglas(prs)
    s_baseline(prs)
    s_scipy_metodos(prs, info)
    s_tps(prs)
    s_perfil(prs)
    s_muestreo(prs, info)
    s_biseccion(prs, info)
    s_loo(prs)
    s_borde(prs)
    s_recon(prs, info)
    s_monotonia(prs, info)
    s_auto(prs)
    s_continuo(prs, info)
    s_fusion(prs)
    s_recorrido(prs)
    s_resultados(prs)
    s_kmc_loop(prs)
    s_kmc_estado(prs)
    s_kmc_tandas(prs, info)
    s_kmc_pasos(prs, info)
    s_preguntas(prs)
    s_cierre(prs)
    prs.save(OUT)
    print("->", OUT, "(%d filminas)" % len(prs.slides))


if __name__ == "__main__":
    main()
