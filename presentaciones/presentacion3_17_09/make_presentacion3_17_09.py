#!/usr/bin/env python
# coding: utf-8
"""Genera presentacion3_17_09.pptx: tercera entrega, reescrita completa el
16/9 siguiendo el mismo patron estructural que presentacion1_02_09 y
presentacion2_09_09 (portada -> que quedo -> como leer cada figura ->
puntos numerados -> conclusiones -> lo que sigue). La version anterior
(borrador armado a los ponchazos durante la sesion del 15/9) quedo en
presentaciones/presentacion3_17_09_v1/.

Cubre, en orden:
  Parte A -- los puntos de la reunion del 9/9:
    1. familia de mascara "grid" (grilla 2D regular, absurdamente uniforme)
       -- reconstrucciones en distintos g, barrido de 5 familias vs %
       observado, conclusiones.
    2. cuantos puntos de cada distribucion caen sobre la frontera.
    3. intento de corregir el escalonado con una media movil sobre la
       curva detectada (resultado negativo/mixto, se prueba honesto).
    4. intento de ventana concentrada en vez de puntos por toda la imagen
       (tampoco gano).
  Parte B -- lo nuevo de seguir investigando esa misma noche:
    5. REG_NOISE_STD=0.08 (el hallazgo grande), validado con frontier_mix
       en 328 y 164 puntos en los 9 g -- "ese es el camino".
  Parte C -- generalizacion:
    6. el mismo comportamiento en imagenes que no son el diagrama de fase.

Uso:  ./venv/bin/python presentaciones/presentacion3_17_09/make_presentacion3_17_09.py
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
OUT_DIR = os.path.join(REPO, "presentaciones", "presentacion3_17_09")
PRES_NAME = "presentacion3_17_09"

R_GSWEEP5 = os.path.join(REPO, "results", "gsweep_5family_summary")
R_FPC = os.path.join(REPO, "results", "frontier_point_count")
R_DIPGF = os.path.join(REPO, "results", "dip_gsweep_frontier")
A_MASKS_G = os.path.join(REPO, "data", "restoration", "masks_g")
R_EXP = os.path.join(REPO, "results", "experimentos_menos_puntos")
R_REG = os.path.join(R_EXP, "barrido_reg_noise_familias")
R_SYNTH_DEMO = os.path.join(REPO, "results", "synthetic_targets_demo")

FAMILIES = ["uniform", "grid", "spread", "frontier", "frontier_mix"]
GS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
MFS = ["0.900", "0.950", "0.980", "0.990", "0.995"]

SLIDE_W, SLIDE_H = 13.33, 7.5
BLUE = RGBColor(0x1F, 0x3B, 0x73)
GREY = RGBColor(0x40, 0x40, 0x40)


# ------------------------------------------------------------ lectura ----------
def gsweep5_table():
    """promedio de PSNR final por (g, familia) sobre los 5 mf."""
    rows = list(csv.DictReader(open(os.path.join(R_GSWEEP5, "summary.csv"))))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["g"], r["family"])].append(float(r["psnr_final"]))
    return {k: sum(v) / len(v) for k, v in agg.items()}


def frontier_frac_table():
    """% de puntos que caen sobre la frontera, promedio/min/max por familia."""
    rows = list(csv.DictReader(open(os.path.join(R_FPC, "summary.csv"))))
    agg = defaultdict(list)
    for r in rows:
        agg[r["family"]].append(float(r["frac_on_edge"]))
    out = {}
    for fam in FAMILIES:
        v = agg[fam]
        out[fam] = (100 * sum(v) / len(v), 100 * min(v), 100 * max(v))
    return out


# --------------------------------------------------------- helpers pptx --------
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


def add_bullets_slide(prs, title, bullets, body_size=17):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, 27)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(12), Inches(5.6))
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
    frac = frontier_frac_table()

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # 1 -- portada
    add_title_slide(
        prs,
        "Reconstruccion del diagrama de fases con Deep Image Prior",
        "Reducir la cantidad de puntos necesarios  ·  tercera entrega  ·  "
        "Santiago Afonso  ·  17/09/2026",
    )

    # 2 -- agenda
    add_bullets_slide(
        prs,
        "Que quedo para esta entrega",
        [
            "De la reunion del 9/9, siempre apuntando a reducir la cantidad de "
            "puntos que hacen falta:",
            "1.  Familia de mascara 'grid' (grilla, absurdamente uniforme): "
            "resultados por g y por cantidad de puntos, y conclusiones.",
            "2.  Cuanto de cada distribucion cae sobre la frontera de fase.",
            "3.  Intento de arreglar el escalonado de la reconstruccion con una "
            "media movil.",
            "4.  Intento de concentrar los puntos en una ventana chica en vez de "
            "repartirlos por toda la imagen.",
            "",
            "Y lo que salio de seguir investigando despues:",
            "5.  Un hallazgo de hiperparametros (REG_NOISE_STD) que resulto ser "
            "el camino mas solido -- validado con 328 y 164 puntos en los 9 g.",
            "6.  Generalizacion: el mismo comportamiento en imagenes que no son "
            "el diagrama de fase.",
        ],
        body_size=16,
    )

    # 3 -- como leer cada figura
    add_full_image_slide(
        prs,
        "Como leer cada figura de resultado",
        os.path.join(R_DIPGF, "g-4.0", "mf0.980", "uniform", "comparison_annotated.png"),
        caption="Original | Enmascarada (lo que ve DIP: solo los puntos observados) | Reconstruida | "
                "|error| (negro = exacto, mas claro = error grande). DIP ajusta la red para que pase "
                "por los puntos observados y el resto lo completa el prior.",
        width=12.8, top=2.5,
    )

    # ============================== PARTE A ==============================

    # 4 -- punto 1: familia grid, que es
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

    # 5 -- punto 1: reconstrucciones reales con grid, distintos g
    s5 = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s5, "1 · Reconstrucciones con la familia grid (2% observado)")
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
        lb = s5.shapes.add_textbox(Inches(left), Inches(top - 0.32), Inches(ex_w), Inches(0.3))
        lp = lb.text_frame.paragraphs[0]
        lp.text = "g = %s -- %s" % (g, desc)
        lp.font.size = Pt(12)
        lp.font.bold = True
        lp.font.color.rgb = BLUE
        iw, ih = _fit(img, ex_w, ex_h_max)
        s5.shapes.add_picture(img, Inches(left), Inches(top), width=Inches(iw), height=Inches(ih))

    # 6 -- punto 1: las 5 distribuciones vs % observado, por g
    add_full_image_slide(
        prs,
        "1 · Las 5 distribuciones en el regimen de g negativo (225 corridas)",
        os.path.join(R_GSWEEP5, "psnr_vs_pct_by_g.png"),
        caption="PSNR final vs. % observado, una por g, 5 lineas (uniform, grid, spread, "
                "frontier, frontier_mix).",
        width=11.8, top=1.35,
    )

    # 7 -- punto 1: tabla resumen + conclusion
    rows = []
    for g in GS:
        gk = str(g)
        rows.append(["%+.1f" % g] + ["%.1f" % avg.get((gk, f), float("nan")) for f in FAMILIES])
    add_table_slide(
        prs,
        "1 · PSNR promedio por g y familia (sobre los 5 mf) -- conclusion",
        ["g"] + FAMILIES,
        rows,
        note="grid (sin ningun conocimiento de la frontera) le gana a frontier_mix (la "
             "estrategia disenada a mano) en la mayoria de los g, sobre todo g >= -2.\n"
             "frontier_mix solo se impone en los g mas duros (-4.0, -3.5). frontier puro "
             "es un desastre en todo el rango (~10-12 dB): sin puntos en las mesetas.",
        col_w=[1.1, 2.05, 2.05, 2.05, 2.05, 2.05], font=12,
    )

    # 8/9 -- ejemplos visuales
    add_full_image_slide(
        prs,
        "1 · Ejemplo: g = -0.5 (frente suave) -- grid le gana por mucho a frontier_mix",
        os.path.join(R_GSWEEP5, "montage_g-0.5_mf0.980.png"),
        caption="2% observado. grid: PSNR 42.3 / SSIM 0.989. frontier_mix: PSNR 28.5 / "
                "SSIM 0.689 -- se ve manchado, textura de ruido lejos de la frontera. "
                "Linea verde = frontera real.",
        width=12.9,
    )
    add_full_image_slide(
        prs,
        "1 · Ejemplo: g = -4.0 (frente abrupto) -- se invierte y aparece el escalonado",
        os.path.join(R_GSWEEP5, "montage_g-4.0_mf0.980.png"),
        caption="2% observado. Aca frontier_mix (22.3 dB) le gana a grid (25.0 dB) pero con "
                "artefacto de escalones. frontier puro colapsa (9.8 dB). Este escalonado es "
                "lo que se intenta corregir en el punto 3.",
        width=12.9,
    )

    # 10 -- punto 2: % de puntos en la frontera por distribucion
    add_table_slide(
        prs,
        "2 · Cuanto de cada distribucion cae sobre la frontera de fase",
        ["familia", "promedio", "minimo", "maximo"],
        [[fam, "%.1f%%" % frac[fam][0], "%.1f%%" % frac[fam][1], "%.1f%%" % frac[fam][2]]
         for fam in FAMILIES],
        note="225 corridas (9 g x 5 mf x 5 familias). frontier concentra en promedio 82%% de "
             "sus puntos sobre la frontera (por diseno) -- y es la peor familia. spread evita "
             "la frontera casi del todo (1%%). grid y uniform, sin buscarlo, ya caen ~6-7%% "
             "sobre la frontera -- suficiente para reconstruir bien.",
        col_w=[3.0, 2.5, 2.5, 2.5], font=13,
    )
    add_full_image_slide(
        prs,
        "2 · Cuantos puntos en la frontera hacen falta para una reconstruccion confiable",
        os.path.join(R_FPC, "psnr_vs_frontier_points.png"),
        caption="Fidelidad final vs. cuantos puntos de la mascara caen sobre la frontera "
                "(no el total observado). Circulo=uniform, triangulo=frontier_mix; color=g.",
        width=12.6, top=1.4,
    )

    # 11 -- punto 3: intento de corregir el escalonado
    add_bullets_slide(
        prs,
        "3 · Intento: arreglar el escalonado con una media movil",
        [
            "La reconstruccion de DIP en frentes abruptos queda con un patron de "
            "escalones (visible en los ejemplos anteriores). Idea: detectar donde "
            "la reconstruccion cruza el nivel 0.5 en cada fila (curva roja), "
            "suavizarla con una media movil (curva verde), y redibujar la imagen "
            "como una linea limpia en esa posicion suavizada.",
            "",
            "Resultado: MIXTO, no un arreglo limpio. En 'grid' el PSNR bajo "
            "(25.0 -> 21.2 dB): la media movil corta camino por dentro de la "
            "curva real cerca del codo de la frontera, gana prolijidad visual y "
            "pierde exactitud. En 'uniform' mejoro apenas (26.2 -> 27.1 dB).",
            "",
            "No se sigue por esta linea -- documentado como probado, no adoptado.",
        ],
        body_size=16,
    )
    add_full_image_slide(
        prs,
        "3 · El resultado: se prolija pero pierde exactitud",
        os.path.join(R_EXP, "correccion_linea", "grid_comparison.png"),
        caption="g=-4.0, grid. Rojo = cruce detectado por fila. Verde = media movil. La "
                "'Corregida' final se ve mas lisa pero el PSNR bajo de 25.0 a 21.2 dB.",
        width=12.6, top=1.5,
    )

    # 12 -- punto 4: intento de ventana concentrada (tecnica sugerida por el profe)
    add_table_slide(
        prs,
        "4 · Intento (sugerido por el director): ventana chica y densa en vez de "
        "puntos por toda la imagen -- sin exito",
        ["Variante", "N pts", "PSNR", "SSIM"],
        [
            ["Global (referencia, grid, 90 pts)", "90", "25.0", "0.963"],
            ["Franja 64px sobre toda la curva", "90", "24.8", "0.935"],
            ["Cuadrado 64x64, pts heredados + reg=0.08", "90", "24.2", "0.957"],
            ["Cuadrado 64x64, pts A PROPOSITO", "121", "22.9", "0.846"],
            ["Cuadrado 32x32 + extrapolacion recta", "36", "18.6", "0.811"],
        ],
        note="Idea: concentrar los pocos puntos en una ventana chica alrededor del codo de "
             "la frontera (reconstruir solo ahi con DIP, empalmar el resto sin tocar) en vez "
             "de repartirlos por toda la imagen. Se probaron 4 variantes en g=-4.0 (franja, "
             "cuadrado 64x64 con puntos heredados o a proposito, cuadrado 32x32 + "
             "extrapolacion) -- NINGUNA le gano al global. Se intento la tecnica tal como se "
             "sugirio y no dio resultado.",
        col_w=[4.8, 1.3, 1.3, 1.3], font=13,
    )

    # ============================== PARTE B ==============================

    add_bullets_slide(
        prs,
        "5 · Lo que salio de seguir investigando: REG_NOISE_STD",
        [
            "Con pocos puntos, DIP sobreajusta al ruido de entrada y deja "
            "'manchas' (textura falsa) en las mesetas -- resulto ser un "
            "problema de HIPERPARAMETRO, no de que distribucion de puntos se "
            "usa.",
            "",
            "Subir REG_NOISE_STD de 0.01 (el que se venia usando para frentes "
            "abruptos) a 0.08 saca esas manchas. Se cruzo con las 3 familias "
            "mas relevantes (grid, uniform, frontier_mix) en 3 g representativos "
            "(-4.0, -2.0, -0.5), 328 puntos.",
            "",
            "Resultado inesperado: frontier_mix -- la familia que en la parte 1 "
            "rendia peor y de forma mas erratica -- es la que MAS se beneficia. "
            "Pasa de ser la peor de las 3 a la mejor por lejos, en los 3 g.",
        ],
        body_size=16,
    )
    add_full_image_slide(
        prs,
        "5 · PSNR por familia, g y REG_NOISE_STD",
        os.path.join(R_REG, "psnr_by_family_g_reg.png"),
        caption="328 pts (2%% obs.). frontier_mix (antes la peor) se vuelve la mejor familia "
                "con reg=0.08 en los 3 g -- saltos de +15 a +24 dB.",
        width=12.6, top=1.4,
    )
    add_full_image_slide(
        prs,
        "5 · Se ve directo en la imagen: las manchas desaparecen",
        os.path.join(R_REG, "reconstrucciones_g_x_reg.png"),
        caption="frontier_mix, 328 pts: filas = g, columnas = REG_NOISE_STD. En reg=0.01 la "
                "meseta negra queda texturada de sobreajuste; en reg=0.08 queda lisa.",
        width=8.2, top=1.35,
    )

    # tabla 328 y 164 puntos, los 9 g -- "ese es el camino"
    def _row_328_164(mf):
        vals = []
        for g in GS:
            p = os.path.join(R_REG, "g%s" % g, "mf%s" % mf, "frontier_mix", "reg0.08", "metrics.csv")
            v = None
            if os.path.isfile(p):
                for row in csv.reader(open(p)):
                    if row and row[0] == "final":
                        v = float(row[1])
            vals.append("%.1f" % v if v is not None else "-")
        return vals

    add_full_image_slide(
        prs,
        "5 · Validado en los 9 g, con dos presupuestos de puntos -- ese es el camino",
        os.path.join(R_REG, "linea_328_vs_164.png"),
        caption="frontier_mix + REG_NOISE_STD=0.08. Con 328 puntos (2%%): 41-52 dB en los 9 "
                "g. Con la MITAD (164 puntos, 1%%): todavia 29-50 dB -- por encima del umbral "
                "confiable en 8 de 9 g. No es un golpe de suerte en un g puntual.",
        width=11.8, top=1.4,
    )
    add_table_slide(
        prs,
        "5 · Los numeros detras del grafico",
        ["g / N"] + ["%+.1f" % g for g in GS],
        [
            ["328 pts (2%%)"] + _row_328_164("0.980"),
            ["164 pts (1%%)"] + _row_328_164("0.990"),
        ],
        col_w=[1.5] + [1.2] * 9, font=11,
    )

    # tres ejemplos visuales, de duro a suave -- no solo el caso mas dificil
    ejemplos_5 = [
        ("-4.0", "el caso mas duro"),
        ("-2.0", "caso intermedio"),
        ("-0.5", "frente suave"),
    ]
    for g, desc in ejemplos_5:
        img = os.path.join(R_REG, "g%s" % g, "mf0.980", "frontier_mix", "reg0.08", "comparison_annotated.png")
        if not os.path.isfile(img):
            continue
        add_full_image_slide(
            prs,
            "5 · g=%s (%s), 328 puntos" % (g, desc),
            img,
            caption="frontier_mix + reg=0.08 -- practicamente indistinguible del original a "
                    "ojo, con solo 328 puntos (2%% observado).",
            width=12.6, top=1.5,
        )

    # los mismos 3 ejemplos, ahora con la MITAD de los puntos (164)
    for g, desc in ejemplos_5:
        img = os.path.join(R_REG, "g%s" % g, "mf0.990", "frontier_mix", "reg0.08", "comparison_annotated.png")
        if not os.path.isfile(img):
            continue
        add_full_image_slide(
            prs,
            "5 · g=%s (%s), con la MITAD de los puntos (164)" % (g, desc),
            img,
            caption="frontier_mix + reg=0.08, 164 puntos (1%% observado, la mitad del "
                    "presupuesto anterior) -- se sostiene fiable, sin cambiar nada mas.",
            width=12.6, top=1.5,
        )

    # hasta donde se puede empujar reg_noise mas alla de 0.08 (g=-4.0, 164 pts)
    push_img = os.path.join(R_EXP, "g4_push_further_164pts", "reg0.12_lr0.001_iter16000",
                             "comparison_annotated.png")
    if os.path.isfile(push_img):
        add_full_image_slide(
            prs,
            "5 · ¿Conviene subir REG_NOISE_STD todavia mas? g=-4.0, 164 puntos",
            push_img,
            caption="reg=0.12 (vs. el 0.08 ya validado), mismo LR e iteraciones, 164 puntos "
                    "(1%% observado): 36.6 dB -- no le gana al 0.08 (38.7 dB con mas "
                    "iteraciones, o 33.6 dB en igualdad de condiciones). No es 'mas ruido "
                    "siempre mejor', hay un optimo cerca de 0.08 en este regimen.",
            width=12.6, top=1.5,
        )

    # cierre fuerte de la parte 5, bien directo para los profesores
    add_bullets_slide(
        prs,
        "5 · Por que este es el camino a seguir",
        [
            "De todo lo probado en esta entrega -- grid, ventanas concentradas, "
            "correccion por media movil -- lo unico que dio un salto grande, "
            "consistente y reproducible en los 9 g fue este cambio de "
            "hiperparametro.",
            "",
            "Funciona con la mitad de los puntos (164) casi tan bien como con el "
            "presupuesto completo (328) -- eso es directamente margen para bajar "
            "el costo de los puntos reales (KMC) sin perder confiabilidad.",
            "",
            "Es ademas el cambio MAS SIMPLE de todos los que se probaron: una "
            "sola linea de configuracion, sin arquitectura nueva, sin "
            "posprocesado, sin ventanas ni puntos inventados.",
            "",
            "Por eso es la direccion en la que se va a seguir profundizando.",
        ],
        body_size=18,
    )

    # ============================== PARTE C ==============================

    add_bullets_slide(
        prs,
        "6 · Generalizando: el mismo comportamiento fuera del diagrama de fase",
        [
            "Para verificar que todo esto no es un capricho particular del "
            "diagrama de fase, se probo DIP con imagenes 'de libro' -- una "
            "campana de Gauss, un barrido de frecuencia, las funciones de "
            "Rosenbrock e Himmelblau, y un tablero de ajedrez -- y con una foto "
            "real (el edificio de FAMAF).",
            "",
            "Mascara Bernoulli uniforme simple (no las distribuciones especiales "
            "armadas para el diagrama de fase), 10%% observado, mismo pipeline.",
            "",
            "La hipotesis: lo que determina si DIP reconstruye bien no es 'es un "
            "diagrama de fase' sino la FRECUENCIA ESPACIAL del contenido -- "
            "conecta directo con Nyquist.",
        ],
        body_size=17,
    )
    add_full_image_slide(
        prs,
        "6 · Cinco funciones matematicas, 10%% observado",
        os.path.join(R_SYNTH_DEMO, "grid_5_matematicas.png"),
        caption="Suave (campana, Rosenbrock, Himmelblau): 46-53 dB, casi perfecto. Alta "
                "frecuencia (tablero, barrido): 20 dB y 7 dB -- confirma la hipotesis, el "
                "problema es la frecuencia del contenido, no el dominio del diagrama de fase.",
        width=13.0, top=1.4,
    )

    famaf_img = os.path.join(R_SYNTH_DEMO, "edificio_famaf", "obs10pct", "comparison_annotated.png")
    if os.path.isfile(famaf_img):
        add_full_image_slide(
            prs,
            "6 · Y con una foto real: el edificio de FAMAF",
            famaf_img,
            caption="1024x768 RGB, 10%% observado, mascara Bernoulli. Mismo pipeline, sin "
                    "ningun ajuste especial para fotos naturales.",
            width=12.6, top=1.5,
        )

    # conclusiones
    add_bullets_slide(
        prs,
        "Conclusiones",
        [
            "grid (sin ningun conocimiento de la frontera) es sorprendentemente "
            "competitiva -- le gana a frontier_mix en la mayoria de los g negativos.",
            "",
            "El escalonado y las ventanas concentradas no se arreglaron con "
            "correcciones post-hoc ni con menos puntos mejor ubicados -- el "
            "camino que si funciono fue un hiperparametro: REG_NOISE_STD=0.08.",
            "",
            "Con ese fix, frontier_mix + 328 puntos da una reconstruccion fiable "
            "(41-52 dB, SSIM >= 0.98) en los 9 g probados -- y se sostiene "
            "razonablemente incluso con la mitad de los puntos (164).",
            "",
            "El comportamiento generaliza: DIP reconstruye bien contenido suave "
            "y falla en contenido de alta frecuencia, sea o no un diagrama de "
            "fase -- confirma que el problema es de muestreo/frecuencia, no "
            "algo especifico de esta fisica.",
        ],
        body_size=16,
    )

    # lo que falta
    add_bullets_slide(
        prs,
        "Lo que falta y lo que sigue",
        [
            "Falta una metrica de error restringida a la banda de frontera (las "
            "metricas actuales son de la imagen completa y las dominan las "
            "mesetas).",
            "",
            "Probar REG_NOISE_STD=0.08 tambien en presupuestos mas chicos que "
            "164 puntos, y en las otras familias (grid, uniform) en los 9 g.",
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
