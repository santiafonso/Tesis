#!/usr/bin/env python
# coding: utf-8
"""Genera presentacion3_13_09.pptx: tercera entrega, sobre los 5 puntos que
dio el director el 9/9 (siempre apuntando a reducir la cantidad de puntos
necesarios para la reconstruccion):

  1. familia de mascara "grid" (grilla 2D regular, determinista)
  2. cuantos puntos caen en la frontera vs. fidelidad de la reconstruccion
  3. correr las 5 distribuciones de muestreo en el regimen de g negativo
  4. overlay de la frontera real (del soc.npy) sobre las reconstrucciones
  5. marco: g negativo es el caso mas comun -> ahi se juega la confiabilidad

Lee de results/ (no versionado, generado por analysis/*.py en esta sesion) y
data/restoration/masks_g/. Mismo estilo y helpers que presentacion2_09_09.

Uso:  ./venv/bin/python presentaciones/presentacion3_13_09/make_presentacion3_13_09.py
"""
import csv
import os
import shutil
from collections import defaultdict

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "presentaciones", "presentacion3_13_09")
PRES_NAME = "presentacion3_13_09"

R_GSWEEP5 = os.path.join(REPO, "results", "gsweep_5family_summary")
R_FPC = os.path.join(REPO, "results", "frontier_point_count")
R_DIPGF = os.path.join(REPO, "results", "dip_gsweep_frontier")
A_MASKS_G = os.path.join(REPO, "data", "restoration", "masks_g")

FAMILIES = ["uniform", "grid", "spread", "frontier", "frontier_mix"]
GS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
MFS = ["0.900", "0.950", "0.980", "0.990", "0.995"]


# ------------------------------------------------------------ lectura ----------
def gsweep5_table():
    """promedio de PSNR final por (g, familia) sobre los 5 mf."""
    rows = list(csv.DictReader(open(os.path.join(R_GSWEEP5, "summary.csv"))))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["g"], r["family"])].append(float(r["psnr_final"]))
    return {k: sum(v) / len(v) for k, v in agg.items()}


def final_row(path):
    if not os.path.isfile(path):
        return None
    psnr = ssim = None
    for row in csv.reader(open(path)):
        if row and row[0] == "final":
            psnr, ssim = float(row[1]), float(row[3])
    return (psnr, ssim) if psnr is not None else None


# --------------------------------------------------------- helpers pptx --------
BLUE = RGBColor(0x1F, 0x3B, 0x73)
GREY = RGBColor(0x40, 0x40, 0x40)
SLIDE_W, SLIDE_H = 13.33, 7.5


def _title(slide, text, size=26):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = BLUE


def add_title_slide(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.7), Inches(2.2), Inches(12), Inches(2))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE
    tb2 = s.shapes.add_textbox(Inches(0.7), Inches(3.7), Inches(12), Inches(1.5))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(19)
    p2.font.color.rgb = GREY
    return s


def add_bullets_slide(prs, title, bullets, body_size=18):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, 27)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(12), Inches(5.4))
    tf = body.text_frame
    tf.word_wrap = True
    for j, b in enumerate(bullets):
        para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        lvl, text = (1, b[2:]) if b.startswith("- ") else (0, b)
        para.text = text
        para.level = lvl
        para.font.size = Pt(body_size if lvl == 0 else body_size - 3)
        para.font.color.rgb = GREY
        para.space_after = Pt(5)
    return s


def _fit(img, max_w, max_h):
    with Image.open(img) as im:
        pw, ph = im.size
    w = float(max_w)
    h = w * ph / pw
    if h > max_h:
        h = float(max_h)
        w = h * pw / ph
    return w, h


def add_full_image_slide(prs, title, img, caption=None, width=None, top=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    band_top = top if top else 1.5
    band_bot = 6.75 if caption else 7.25
    max_w = float(width) if width else 12.6
    max_h = band_bot - band_top
    w, h = _fit(img, max_w, max_h)
    left = (SLIDE_W - w) / 2
    t = band_top + (max_h - h) / 2
    s.shapes.add_picture(img, Inches(left), Inches(t), width=Inches(w), height=Inches(h))
    if caption:
        cb = s.shapes.add_textbox(Inches(0.6), Inches(band_bot + 0.05), Inches(12.1), Inches(0.6))
        cp = cb.text_frame.paragraphs[0]
        cp.text = caption
        cp.font.size = Pt(12)
        cp.font.italic = True
        cp.font.color.rgb = GREY
        cp.alignment = PP_ALIGN.CENTER
    return s


def add_table_slide(prs, title, headers, rows, note=None, left=1.0, col_w=None, font=12):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    nr, nc = len(rows) + 1, len(headers)
    total_w = 11.3
    tbl = s.shapes.add_table(nr, nc, Inches(left), Inches(1.5),
                              Inches(total_w), Inches(min(0.42 * nr, 5.2))).table
    if col_w:
        for c, w in enumerate(col_w):
            tbl.columns[c].width = Inches(w)
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = h
        pr = cell.text_frame.paragraphs[0]
        pr.font.size = Pt(font + 1)
        pr.font.bold = True
    for r, row in enumerate(rows, start=1):
        for c, v in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(v)
            pr = cell.text_frame.paragraphs[0]
            pr.font.size = Pt(font)
            pr.font.bold = (c == 0)
    if note:
        nb = s.shapes.add_textbox(Inches(left), Inches(1.5 + min(0.42 * nr, 5.2) + 0.2),
                                   Inches(total_w), Inches(1.2))
        tf = nb.text_frame
        tf.word_wrap = True
        for j, line in enumerate(note.split("\n")):
            para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            para.text = line
            para.font.size = Pt(11)
            para.font.italic = True
            para.font.color.rgb = GREY
    return s


# ----------------------------------------------------------------- build -------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    avg = gsweep5_table()

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # 1 -- portada
    add_title_slide(
        prs,
        "Reconstruccion del diagrama de fases con Deep Image Prior",
        "Reducir la cantidad de puntos necesarios  ·  tercera entrega  ·  "
        "Santiago Afonso  ·  16/09/2026",
    )

    # 2 -- que quedo para esta entrega
    add_bullets_slide(
        prs,
        "Que quedo de la reunion del 9/9",
        [
            "Cinco cosas, siempre apuntando a reducir la cantidad de puntos que hacen "
            "falta para reconstruir el mapa:",
            "1.  Una distribucion de muestreo 'absurdamente uniforme': grilla 2D regular, "
            "sin azar.",
            "2.  Contar cuantos puntos de una mascara uniforme caen en la frontera, y "
            "cruzarlo con que tan buena sale la reconstruccion.",
            "3.  Correr las distintas distribuciones de muestreo en el regimen de g "
            "negativo.",
            "4.  Trazar la frontera real (conocida, del modelo del continuo) sobre las "
            "reconstrucciones, para ver donde se desvian.",
            "5.  g negativo es el caso mas comun/real -> ahi se juega la confiabilidad.",
        ],
        body_size=18,
    )

    # 3 -- punto 1: familia grid
    preview = os.path.join(A_MASKS_G, "g0.0", "mf0.980", "preview.png")
    add_full_image_slide(
        prs,
        "1 · Una familia de mascara nueva: grid",
        preview,
        caption="grid: grilla 2D regular de paso fijo (round(sqrt(H*W/N)) por eje), sin "
                "azar -- ninguna informacion sobre donde esta la frontera. Mismo N que "
                "las otras 4 familias para cada fraccion observada.",
        width=12.6,
    )

    # 3b -- punto 1: reconstrucciones reales hechas con grid, distintos g
    s3b = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s3b, "1 · Reconstrucciones con la familia grid (2% observado)")
    grid_examples = [
        ("-4.0", "frente abrupto (el caso mas dificil)"),
        ("-2.0", "frente intermedio"),
        ("-1.0", "frente suave"),
        ("0.0", "sin interaccion (g=0)"),
    ]
    ex_w, ex_h_max = 6.0, 1.75
    cols_left = [0.55, 6.85]
    rows_top = [1.75, 3.95]
    for i, (g, desc) in enumerate(grid_examples):
        img = os.path.join(R_DIPGF, "g%s" % g, "mf0.980", "grid", "comparison_annotated.png")
        if not os.path.isfile(img):
            continue
        left = cols_left[i % 2]
        top = rows_top[i // 2]
        lb = s3b.shapes.add_textbox(Inches(left), Inches(top - 0.32), Inches(ex_w), Inches(0.3))
        lp = lb.text_frame.paragraphs[0]
        lp.text = "g = %s -- %s" % (g, desc)
        lp.font.size = Pt(12)
        lp.font.bold = True
        lp.font.color.rgb = BLUE
        iw, ih = _fit(img, ex_w, ex_h_max)
        s3b.shapes.add_picture(img, Inches(left), Inches(top), width=Inches(iw), height=Inches(ih))

    # 4 -- punto 2: puntos en frontera vs fidelidad
    add_full_image_slide(
        prs,
        "2 · Cuantos puntos en la frontera hacen falta para una reconstruccion confiable",
        os.path.join(R_FPC, "psnr_vs_frontier_points.png"),
        caption="Fidelidad final vs. cuantos puntos de la mascara caen sobre la frontera "
                "(no el total observado). Circulo=uniform, triangulo=frontier_mix; color=g. "
                "225 corridas.",
        width=12.6, top=1.4,
    )

    # 5 -- punto 3: la grilla 3x3 + dos imagenes reales de ejemplo
    R_DIPGF_G = lambda g, mf, fam: os.path.join(  # noqa: E731
        R_DIPGF, "g%s" % g, "mf%s" % mf, fam, "comparison_annotated.png")
    s5 = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s5, "3 · Las 5 distribuciones en el regimen de g negativo (225 corridas)")
    plot_path = os.path.join(R_GSWEEP5, "psnr_vs_pct_by_g.png")
    pw, ph = _fit(plot_path, 7.6, 5.55)
    s5.shapes.add_picture(plot_path, Inches(0.35), Inches(1.35), width=Inches(pw), height=Inches(ph))
    cap5 = s5.shapes.add_textbox(Inches(0.35), Inches(1.35 + ph + 0.05), Inches(7.6), Inches(0.5))
    cp5 = cap5.text_frame.paragraphs[0]
    cp5.text = "PSNR final vs. % observado, una por g, 5 lineas (uniform, grid, spread, frontier, frontier_mix)."
    cp5.font.size = Pt(11)
    cp5.font.italic = True
    cp5.font.color.rgb = GREY
    # dos imagenes reales al lado (mismo caso duro g=-4.0, mf=0.980 -- 2% obs
    # -- donde el grafico muestra a frontier_mix cruzando por encima de grid)
    ex_left = 8.15
    ex_w = 4.85
    for i, (fam, label) in enumerate([("grid", "grid"), ("frontier_mix", "frontier_mix")]):
        img = R_DIPGF_G("-4.0", "0.980", fam)
        if not os.path.isfile(img):
            continue
        iw, ih = _fit(img, ex_w, 2.55)
        top = 1.75 + i * 2.85
        s5.shapes.add_picture(img, Inches(ex_left), Inches(top), width=Inches(iw), height=Inches(ih))
        lb = s5.shapes.add_textbox(Inches(ex_left), Inches(top - 0.32), Inches(ex_w), Inches(0.3))
        lp = lb.text_frame.paragraphs[0]
        lp.text = "g=-4.0, 2%% obs. -- %s" % label
        lp.font.size = Pt(11)
        lp.font.bold = True
        lp.font.color.rgb = BLUE

    # 6 -- punto 3: tabla resumen
    rows = []
    for g in GS:
        gk = str(g)
        rows.append(["%+.1f" % g] + ["%.1f" % avg.get((gk, f), float("nan")) for f in FAMILIES])
    add_table_slide(
        prs,
        "3 · PSNR promedio por g y familia (sobre los 5 mf)",
        ["g"] + FAMILIES,
        rows,
        note="frontier puro es un desastre en todo el rango (~10-12 dB): concentrar "
             "todo en la frontera deja las mesetas sin puntos.\n"
             "grid (sin ningun conocimiento de la frontera) le gana a frontier_mix (la "
             "estrategia disenada a mano) en la mayoria de los g, sobre todo g >= -2. "
             "frontier_mix solo se impone en los g mas duros (-4.0, -3.5).",
        col_w=[1.1, 2.05, 2.05, 2.05, 2.05, 2.05], font=12,
    )

    # 7/8 -- ejemplos visuales del punto 3
    add_full_image_slide(
        prs,
        "3 · Ejemplo: g = -0.5 (frente suave) -- grid le gana por mucho a frontier_mix",
        os.path.join(R_GSWEEP5, "montage_g-0.5_mf0.980.png"),
        caption="2% observado (mf=0.980). grid: PSNR 42.3 / SSIM 0.989. frontier_mix: "
                "PSNR 28.5 / SSIM 0.689 -- se ve manchado, textura de ruido lejos de la "
                "frontera.",
        width=12.9,
    )
    add_full_image_slide(
        prs,
        "3 · Ejemplo: g = -4.0 (frente abrupto, el caso dificil) -- se invierte",
        os.path.join(R_GSWEEP5, "montage_g-4.0_mf0.980.png"),
        caption="2% observado. Aca frontier_mix (22.3 dB) le gana a grid (25.0 dB en "
                "PSNR pero con artefacto de escalones -- ver punto 4). frontier puro "
                "colapsa (9.8 dB): sin puntos en las mesetas, DIP alucina textura ahi.",
        width=12.9,
    )

    # 9 -- punto 4: overlay de la frontera real
    add_bullets_slide(
        prs,
        "4 · Overlay de la frontera real sobre las reconstrucciones",
        [
            "La frontera real se calcula del soc.npy denso (el modelo del continuo, no "
            "hace falta 'a mano' con el mouse): mismo metodo que dip.frontier_mask "
            "(|grad SoC| suavizado, contorno al percentil 90).",
            "",
            "Se superpone esa curva (linea verde) sobre la reconstruccion de cada "
            "familia, para ver a ojo donde el DIP se desvia de la posicion real del "
            "frente -- diagnostico visual, no una metrica numerica nueva.",
            "",
            "IMPORTANTE: esto es solo el overlay/diagnostico. Falta ver si conviene ir "
            "un paso mas -- usar esa frontera conocida para corregir/suavizar la "
            "reconstruccion (no solo mostrarla encima). Eso todavia no esta hecho.",
        ],
        body_size=17,
    )

    # 10/11 -- overlay: ejemplos
    add_full_image_slide(
        prs,
        "4 · g = -1.0 (frente moderado): la reconstruccion sigue la frontera real casi perfecto",
        os.path.join(R_GSWEEP5, "montage_g-1.0_mf0.980.png"),
        caption="Linea verde = frontera real. uniform, grid y frontier_mix pegan a la "
                "curva en todo el recorrido -- desviacion practicamente invisible a esta "
                "escala.",
        width=12.9,
    )
    add_full_image_slide(
        prs,
        "4 · g = -4.0 (frente abrupto): aparece un patron de escalones",
        os.path.join(R_GSWEEP5, "montage_g-4.0_mf0.980.png"),
        caption="Comparando con la linea verde: incluso las familias 'buenas' (uniform, "
                "grid, spread) meten un dentado en vez de la diagonal lisa real. "
                "frontier_mix es la que menos dientes muestra en este regimen.",
        width=12.9,
    )

    # 12 -- conclusiones
    add_bullets_slide(
        prs,
        "Conclusiones",
        [
            "grid (una grilla regular, sin ningun conocimiento de la frontera) es "
            "sorprendentemente competitiva -- le gana a la estrategia disenada a mano "
            "(frontier_mix) en la mayoria de los regimenes de g negativo.",
            "",
            "frontier_mix solo se justifica en los casos mas duros (g <= -3.5) y con "
            "densidad de puntos moderada -- no es una mejora universal.",
            "",
            "Concentrar puntos solo en la frontera (familia frontier) sigue siendo lo "
            "peor que se puede hacer, en cualquier g: las mesetas quedan sin fijar.",
            "",
            "El overlay confirma un patron nuevo: en frentes abruptos, incluso las "
            "reconstrucciones buenas por PSNR/SSIM muestran un artefacto de escalones "
            "que esas metricas no capturan bien.",
        ],
        body_size=17,
    )

    # 13 -- lo que falta
    add_bullets_slide(
        prs,
        "Lo que falta y lo que sigue",
        [
            "Punto 4, a confirmar: si la idea era ademas CORREGIR la reconstruccion "
            "usando la frontera conocida (no solo mostrarla encima), eso todavia no "
            "esta implementado.",
            "",
            "Falta una metrica de error restringida a la banda de frontera (las metricas "
            "actuales son de la imagen completa y las dominan las mesetas).",
            "",
            "Pasar de los mapas del continuo a puntos KMC reales sigue pendiente.",
        ],
        body_size=17,
    )

    # 14 -- extra (fuera de los 5 puntos): ventanas concentradas
    R_EXP = os.path.join(REPO, "results", "experimentos_menos_puntos")
    add_bullets_slide(
        prs,
        "Extra (fuera de los 5 puntos): ¿ventana chica y densa en vez de "
        "puntos por toda la imagen?",
        [
            "Idea explorada en la misma sesion (no pedida por el director): en vez "
            "de repartir pocos puntos por toda la imagen, concentrarlos en una "
            "ventana chica alrededor del codo de la frontera -- reconstruir solo "
            "ahi con DIP y empalmar el resto de la imagen sin tocar.",
            "",
            "Se probaron 4 variantes en g=-4.0 (el caso mas duro): una franja de "
            "64px siguiendo toda la curva, un cuadrado 64x64 con los puntos que "
            "le tocaban de la mascara grid global, el mismo cuadrado con puntos "
            "generados a proposito para esa ventana, y un cuadrado 32x32 real "
            "con extrapolacion de la recta fuera de la ventana (turno anterior "
            "de hoy: la extrapolacion asume una recta, pero la frontera real se "
            "curva en meseta vertical/horizontal fuera de la zona angosta).",
        ],
        body_size=16,
    )

    add_table_slide(
        prs,
        "Extra: PSNR/SSIM de cada variante (imagen completa, g=-4.0)",
        ["Variante", "N pts", "PSNR", "SSIM"],
        [
            ["Global (referencia, grid, 90 pts)", "90", "25.0", "0.963"],
            ["Franja 64px sobre toda la curva", "90", "24.8", "0.935"],
            ["Cuadrado 64x64, pts heredados", "90", "24.2", "0.863"],
            ["  + REG_NOISE_STD=0.08", "90", "24.2", "0.957"],
            ["Cuadrado 64x64, pts A PROPOSITO", "121", "22.9", "0.846"],
            ["Cuadrado 32x32 + extrapolacion recta", "36", "18.6", "0.811"],
        ],
        note="Ninguna variante le gana claramente al global -- y ninguna le gana "
             "al hallazgo mas simple de la sesion (subir REG_NOISE_STD a 0.08 en "
             "la imagen COMPLETA, sin ventana, ya lleva a grid/120pts a 27.2 dB).\n"
             "Achicar la ventana (32x32) o extrapolar mas alla de ella empeora: "
             "la frontera real no es una recta fuera de la zona angosta.",
        col_w=[4.6, 1.3, 1.3, 1.3], font=13,
    )

    s14c = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s14c, "Extra: como se ve -- ventana concentrada vs. extrapolacion")
    img_a = os.path.join(R_EXP, "ventana_cuadrado_64", "window_comparison_sq64_reg0.08.png")
    img_b = os.path.join(R_EXP, "extrapolacion", "quadrant_extrapolate_comparison_ultimo_32x32.png")
    wa, ha = _fit(img_a, 12.4, 2.9)
    s14c.shapes.add_picture(img_a, Inches((SLIDE_W - wa) / 2), Inches(1.35), width=Inches(wa), height=Inches(ha))
    cap_a = s14c.shapes.add_textbox(Inches(0.6), Inches(1.35 + ha + 0.05), Inches(12.1), Inches(0.35))
    cap_a.text_frame.paragraphs[0].text = (
        "Cuadrado 64x64 con los puntos heredados de la mascara grid + REG_NOISE_STD=0.08: PSNR 24.2, SSIM 0.957 -- practicamente empata al global.")
    cap_a.text_frame.paragraphs[0].font.size = Pt(11)
    cap_a.text_frame.paragraphs[0].font.italic = True
    cap_a.text_frame.paragraphs[0].font.color.rgb = GREY
    top_b = 1.35 + ha + 0.45
    wb, hb = _fit(img_b, 12.4, 7.15 - top_b - 0.35)
    s14c.shapes.add_picture(img_b, Inches((SLIDE_W - wb) / 2), Inches(top_b), width=Inches(wb), height=Inches(hb))
    cap_b = s14c.shapes.add_textbox(Inches(0.6), Inches(top_b + hb + 0.05), Inches(12.1), Inches(0.35))
    cap_b.text_frame.paragraphs[0].text = (
        "Cuadrado 32x32 (36 pts) + recta extrapolada fuera de la ventana: PSNR 18.6, SSIM 0.811 -- la recta no sigue la curvatura real de la meseta.")
    cap_b.text_frame.paragraphs[0].font.size = Pt(11)
    cap_b.text_frame.paragraphs[0].font.italic = True
    cap_b.text_frame.paragraphs[0].font.color.rgb = GREY

    # 17 -- extra: REG_NOISE_STD x familia x g (corrido en cluster, 15/9)
    R_REGFAM = os.path.join(REPO, "results", "experimentos_menos_puntos", "barrido_reg_noise_familias")
    add_bullets_slide(
        prs,
        "Extra: REG_NOISE_STD en las 3 mejores familias, en 3 g (cluster)",
        [
            "El hallazgo de REG_NOISE_STD=0.08 (arregla las 'manchas' de "
            "sobreajuste con pocos puntos) se probo hasta ahora solo en grid. "
            "Se extendio a las 3 familias mas relevantes (grid, uniform, "
            "frontier_mix) en 3 g representativos (-4.0 duro, -2.0 medio, -0.5 "
            "suave), 2% observado -- 27 corridas en el cluster.",
            "",
            "Resultado inesperado: frontier_mix (la familia que peor y mas "
            "erraticamente rendia) es la que MAS se beneficia -- pasa de ser la "
            "peor de las 3 a la mejor por lejos con reg=0.08, en los 3 g "
            "(+15 a +24 dB de salto). grid y uniform mejoran mucho menos, y "
            "grid incluso empeora un poco en el g mas duro (-4.0).",
        ],
        body_size=17,
    )

    add_full_image_slide(
        prs,
        "Extra: PSNR por familia, g y REG_NOISE_STD",
        os.path.join(R_REGFAM, "psnr_by_family_g_reg.png"),
        caption="mf=0.980 (~328 pts). frontier_mix (antes la peor) se vuelve la mejor familia con reg=0.08 en los 3 g.",
        width=12.6, top=1.4,
    )

    s17c = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s17c, "Extra: frontier_mix en g=-0.5 -- antes y despues de reg=0.08")
    img_c = os.path.join(R_REGFAM, "g-0.5", "mf0.980", "frontier_mix", "reg0.01", "comparison_annotated.png")
    img_d = os.path.join(R_REGFAM, "g-0.5", "mf0.980", "frontier_mix", "reg0.08", "comparison_annotated.png")
    wc, hc = _fit(img_c, 12.4, 2.7)
    s17c.shapes.add_picture(img_c, Inches((SLIDE_W - wc) / 2), Inches(1.35), width=Inches(wc), height=Inches(hc))
    cap_c = s17c.shapes.add_textbox(Inches(0.6), Inches(1.35 + hc + 0.05), Inches(12.1), Inches(0.3))
    cap_c.text_frame.paragraphs[0].text = "reg=0.01 (default anterior): PSNR 26.7, SSIM 0.711 -- manchas de sobreajuste en la meseta negra."
    cap_c.text_frame.paragraphs[0].font.size = Pt(11)
    cap_c.text_frame.paragraphs[0].font.italic = True
    cap_c.text_frame.paragraphs[0].font.color.rgb = GREY
    top_d = 1.35 + hc + 0.45
    wd, hd = _fit(img_d, 12.4, 7.15 - top_d - 0.3)
    s17c.shapes.add_picture(img_d, Inches((SLIDE_W - wd) / 2), Inches(top_d), width=Inches(wd), height=Inches(hd))
    cap_d = s17c.shapes.add_textbox(Inches(0.6), Inches(top_d + hd + 0.05), Inches(12.1), Inches(0.3))
    cap_d.text_frame.paragraphs[0].text = "reg=0.08: PSNR 51.1, SSIM 0.997 -- +24 dB, meseta limpia, error solo pegado al borde."
    cap_d.text_frame.paragraphs[0].font.size = Pt(11)
    cap_d.text_frame.paragraphs[0].font.italic = True
    cap_d.text_frame.paragraphs[0].font.color.rgb = GREY

    # 18 -- extra: barrido de LR (aparte de NUM_ITER/REG_NOISE_STD)
    add_table_slide(
        prs,
        "Extra: tocar el learning rate (LR) en el regimen de MUY pocos puntos",
        ["LR", "REG_NOISE_STD", "PSNR", "SSIM"],
        [
            ["0.001 (default)", "0.01 (default)", "19.3", "0.609"],
            ["0.0005", "0.01", "18.3", "0.567"],
            ["0.0005", "0.005", "17.6", "0.567"],
            ["0.001", "0.005", "20.3", "0.685"],
            ["0.002", "0.01", "20.1", "0.828"],
            ["0.001", "0.03", "19.8", "0.868"],
        ],
        note="g=-4.0, grid, mf=0.995 (~82 pts, el caso mas disperso corrido hasta ahora).\n"
             "El 'combo clasico' del paper de DIP (LR bajo + ruido bajo) fue lo PEOR de "
             "todo lo probado. Lo que si ayuda: subir el ruido de regularizacion (misma "
             "direccion que el hallazgo de reg=0.08) o subir un poco el LR -- ambos "
             "mejoran PSNR y SSIM sin agregar puntos.",
        col_w=[2.3, 2.3, 1.5, 1.5], font=13,
    )

    out = os.path.join(OUT_DIR, PRES_NAME + ".pptx")
    prs.save(out)
    snap = os.path.join(OUT_DIR, "make_" + PRES_NAME + ".py")
    if os.path.abspath(__file__) != os.path.abspath(snap):
        shutil.copy(__file__, snap)
    print("Presentacion guardada en:", out, "(%d diapositivas)" % len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
