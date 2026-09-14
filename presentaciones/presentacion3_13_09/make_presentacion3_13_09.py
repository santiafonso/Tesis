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

    # 5 -- punto 3: la grilla 3x3
    add_full_image_slide(
        prs,
        "3 · Las 5 distribuciones en el regimen de g negativo (225 corridas)",
        os.path.join(R_GSWEEP5, "psnr_vs_pct_by_g.png"),
        caption="PSNR final vs. % observado, una diapositiva por g, 5 lineas (uniform, "
                "grid, spread, frontier, frontier_mix).",
        width=11.5, top=1.3,
    )

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

    out = os.path.join(OUT_DIR, PRES_NAME + ".pptx")
    prs.save(out)
    snap = os.path.join(OUT_DIR, "make_" + PRES_NAME + ".py")
    if os.path.abspath(__file__) != os.path.abspath(snap):
        shutil.copy(__file__, snap)
    print("Presentacion guardada en:", out, "(%d diapositivas)" % len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
