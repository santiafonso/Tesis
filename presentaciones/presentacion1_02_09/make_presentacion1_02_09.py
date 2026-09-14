#!/usr/bin/env python
# coding: utf-8
"""Genera presentacion_DIP_diagrama_fases.pptx a partir de los resultados de los
barridos de mascara (results_gris_64/ y results_gris_128/) y el diagrama del
modelo del continuo. Tambien produce la figura resumen sweep_summary.png.

Uso:  ./venv/bin/python make_presentacion.py
"""
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ------------------------------------------------------------- salida ----------
# Cada presentacion vive en su propia carpeta con fecha: presentacionN_DD_MM.
PRES_NAME = os.environ.get("PRES_NAME", "presentacion1_02_09")
OUT_DIR = os.path.join("presentaciones", PRES_NAME)

# ---------------------------------------------------------------- datos ---------
MASKS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]

# (psnr, ssim, mae) finales -- copiados de results_gris_{64,128}/mf*/metrics.csv
DATA = {
    64: {
        0.50: (51.07, 0.9988, 0.00106),
        0.70: (47.17, 0.9983, 0.00124),
        0.80: (47.44, 0.9979, 0.00160),
        0.90: (40.49, 0.9914, 0.00232),
        0.95: (36.28, 0.9892, 0.00413),
        0.98: (29.74, 0.9473, 0.00965),
        0.99: (23.96, 0.8482, 0.02255),
    },
    128: {
        0.50: (55.71, 0.9989, 0.00091),
        0.70: (52.21, 0.9968, 0.00142),
        0.80: (54.38, 0.9990, 0.00096),
        0.90: (52.09, 0.9985, 0.00111),
        0.95: (44.81, 0.9962, 0.00152),
        0.98: (38.92, 0.9884, 0.00276),
        0.99: (33.30, 0.9628, 0.00671),
    },
}


def visibles(mf, lado):
    return int(round((1 - mf) * lado * lado))


# ---------------------------------------------------- figura resumen -----------
def fig_resumen(path):
    x = [m * 100 for m in MASKS]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for res, marker in [(64, "o-"), (128, "s-")]:
        psnr = [DATA[res][m][0] for m in MASKS]
        ssim = [DATA[res][m][1] for m in MASKS]
        mae = [DATA[res][m][2] * 100 for m in MASKS]  # puntos porcentuales de SoC
        axs[0].plot(x, psnr, marker, label="%dx%d" % (res, res))
        axs[1].plot(x, ssim, marker, label="%dx%d" % (res, res))
        axs[2].plot(x, mae, marker, label="%dx%d" % (res, res))
    axs[0].set_title("PSNR  (mas alto = mejor)")
    axs[0].set_ylabel("dB")
    axs[1].set_title("SSIM  (1 = identico)")
    axs[1].set_ylim(0.80, 1.005)
    axs[2].set_title("MAE  (mas bajo = mejor)")
    axs[2].set_ylabel("puntos porcentuales de SoC")
    for ax in axs:
        ax.set_xlabel("% de puntos ocultos")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura resumen:", path)


# --------------------------------------------------------- helpers pptx --------
BLUE = RGBColor(0x1F, 0x3B, 0x73)
GREY = RGBColor(0x40, 0x40, 0x40)
CODEBG = RGBColor(0xF3, 0xF3, 0xF3)


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
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = BLUE
    tb2 = s.shapes.add_textbox(Inches(0.7), Inches(3.6), Inches(12), Inches(1.5))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(20)
    p2.font.color.rgb = GREY
    return s


def add_bullets_slide(prs, title, bullets, img=None, img_left=7.3, img_top=1.6,
                      img_width=5.6, note=None, body_size=18):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, 28)
    body_w = 12 if img is None else (img_left - 0.9)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(body_w), Inches(4.9))
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
    if img:
        s.shapes.add_picture(img, Inches(img_left), Inches(img_top),
                             width=Inches(img_width))
    if note:
        nb = s.shapes.add_textbox(Inches(0.6), Inches(6.35), Inches(12.2), Inches(1.0))
        tf2 = nb.text_frame
        tf2.word_wrap = True
        for j, line in enumerate(note.split("\n")):
            para = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
            para.text = line
            para.font.size = Pt(10)
            para.font.italic = True
            para.font.color.rgb = GREY
    return s


def add_full_image_slide(prs, title, img, caption=None, width=12.6, top=2.2):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    s.shapes.add_picture(img, Inches((13.33 - width) / 2), Inches(top),
                         width=Inches(width))
    if caption:
        cb = s.shapes.add_textbox(Inches(0.6), Inches(6.75), Inches(12.1), Inches(0.6))
        cp = cb.text_frame.paragraphs[0]
        cp.text = caption
        cp.font.size = Pt(13)
        cp.font.italic = True
        cp.font.color.rgb = GREY
        cp.alignment = PP_ALIGN.CENTER
    return s


def add_code_slide(prs, title, code, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(12), Inches(4.9))
    box.fill.solid()
    box.fill.fore_color.rgb = CODEBG
    tf = box.text_frame
    tf.word_wrap = True
    for j, line in enumerate(code.split("\n")):
        para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        para.text = line
        para.font.name = "Consolas"
        para.font.size = Pt(13)
        para.font.color.rgb = GREY
    if note:
        nb = s.shapes.add_textbox(Inches(0.7), Inches(6.5), Inches(12), Inches(0.8))
        npar = nb.text_frame.paragraphs[0]
        npar.text = note
        npar.font.size = Pt(13)
        npar.font.italic = True
        npar.font.color.rgb = GREY
    return s


def add_params_slide(prs, title, pairs, img=None, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    rows = len(pairs)
    tbl_w = 6.4 if img else 9.0
    tbl = s.shapes.add_table(rows, 2, Inches(0.7), Inches(1.5),
                             Inches(tbl_w), Inches(0.4 * rows)).table
    for r, (k, v) in enumerate(pairs):
        for c, val in enumerate((k, v)):
            cell = tbl.cell(r, c)
            cell.text = val
            pr = cell.text_frame.paragraphs[0]
            pr.font.size = Pt(12)
            pr.font.bold = (c == 0)
    if img:
        s.shapes.add_picture(img, Inches(7.4), Inches(1.5), width=Inches(5.4))
    if note:
        nb = s.shapes.add_textbox(Inches(0.7), Inches(1.5 + 0.4 * rows + 0.15),
                                  Inches(12), Inches(1.4))
        tf = nb.text_frame
        tf.word_wrap = True
        for j, line in enumerate(note.split("\n")):
            para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            para.text = line
            para.font.size = Pt(10)
            para.font.italic = True
            para.font.color.rgb = GREY
    return s


def add_table_slide(prs, title, res):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    rows = len(MASKS) + 1
    tbl = s.shapes.add_table(rows, 5, Inches(1.6), Inches(1.6),
                             Inches(10), Inches(0.42 * rows)).table
    for c, h in enumerate(["% oculto", "puntos visibles", "PSNR [dB]", "SSIM",
                           "MAE [SoC]"]):
        cell = tbl.cell(0, c)
        cell.text = h
        cell.text_frame.paragraphs[0].font.size = Pt(14)
        cell.text_frame.paragraphs[0].font.bold = True
    for r, mf in enumerate(MASKS, start=1):
        psnr, ssim, mae = DATA[res][mf]
        vals = ["%d%%" % (mf * 100), str(visibles(mf, res)), "%.1f" % psnr,
                "%.4f" % ssim, "%.4f" % mae]
        for c, v in enumerate(vals):
            cell = tbl.cell(r, c)
            cell.text = v
            cell.text_frame.paragraphs[0].font.size = Pt(13)
    nb = s.shapes.add_textbox(Inches(1.6), Inches(1.6 + 0.42 * rows + 0.25),
                              Inches(10), Inches(0.8))
    np_ = nb.text_frame.paragraphs[0]
    np_.text = ("Imagen %dx%d = %d puntos en total.  'puntos visibles' = los que ve "
                "DIP; el resto los reconstruye." % (res, res, res * res))
    np_.font.size = Pt(13)
    np_.font.italic = True
    np_.font.color.rgb = GREY
    return s


# ----------------------------------------------------------------- build -------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    resumen_png = os.path.join(OUT_DIR, "sweep_summary.png")
    fig_resumen(resumen_png)

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    R = "results_gris_128"

    # 1 -- portada
    add_title_slide(
        prs,
        "Reconstruccion del diagrama de fases con Deep Image Prior",
        "Interpolacion de mapas SoC_max desde muestreo disperso  ·  Santiago Afonso",
    )

    # 2 -- el problema (breve)
    add_bullets_slide(
        prs,
        "El problema",
        [
            "El diagrama de fases SoC_max(log Xi, log l) se calcula punto por punto.",
            "Cada punto 'real' es una corrida KMC costosa: una malla densa es inviable.",
            "Idea: correr pocos puntos y reconstruir el mapa completo (Deep Image Prior).",
            "Pregunta de esta etapa: cuantos puntos hacen falta para una reconstruccion fiel.",
        ],
    )

    # 3 -- parametros del modelo del continuo
    add_params_slide(
        prs,
        "Con que parametros se genero el mapa del continuo",
        [
            ("Script", "generate_phase_diagram.py  (NUM_XI = NUM_ELL = 128)"),
            ("Paquete", "galpynostatic 0.5.13"),
            ("Modelo", "single-particle + Butler-Volmer, isoterma de Langmuir"),
            ("vcut (phi_cut)", "-0.15"),
            ("g (interaccion)", "0.0"),
            ("Rango log(Xi)", "-4.0  ...  2.0"),
            ("Rango log(l)", "-4.0  ...  2.0"),
            ("Puntos de malla", "128 x 128"),
            ("grid_size", "1000"),
            ("time_steps", "100000"),
            ("nthreads", "-1  (todos los nucleos)"),
        ],
        img="results_phase_diagram/phase_diagram_reference_128.png",
        note=(
            "Fuente de los parametros:\n"
            "- Modelo (SPM + Butler-Volmer + Langmuir-Frumkin), parametros Xi / l, "
            "grid_size, time_steps y resolucion 128x128:\n"
            "  Ruderman et al., Phys. Scr. 100 (2025) 015946  (paper del software "
            "galpynostatic).\n"
            "- vcut = -0.15 V (potencial de corte) y g = 0.0 (isoterma de Langmuir):\n"
            "  Gavilan-Arriazu et al., Entropy 27 (2025) 663  -- el mapa reproducido es "
            "su Figura 11f.\n"
            "- Marco del mapa (Xi, l): ChemPhysChem 24 (2023) e202200665  y  "
            "Electrochim. Acta 523 (2025) 145939."
        ),
    )

    # 4 -- que es cada metrica (simple) + de donde salio cada una
    add_bullets_slide(
        prs,
        "Que es el MAE, el PSNR y el SSIM",
        [
            "MAE (error absoluto medio): el error promedio de SoC, en las mismas unidades",
            "- que el SoC. MAE = 0.001  ->  se equivoca en promedio 0.1 puntos porcentuales.",
            "",
            "PSNR (relacion senal-ruido de pico): el mismo error en escala logaritmica de",
            "- decibeles. Mas alto = mejor. Arriba de 40 dB la diferencia es casi invisible.",
            "",
            "SSIM (similitud estructural): compara la ESTRUCTURA local (bordes, contraste,",
            "- formas), no pixel por pixel. Va de 0 a 1; 1 = identico. Verifica que la",
            "- frontera entre las mesetas quede en el lugar correcto.",
        ],
        body_size=16,
        note=(
            "De donde salio cada una:\n"
            "- PSNR: heredada del paper de Deep Image Prior (Ulyanov et al., CVPR 2018); "
            "es la unica metrica cuantitativa que usan (denoising, inpainting, "
            "super-resolucion). El codigo original ya la trackea y la usa para el "
            "backtracking. Se calcula con skimage.metrics.peak_signal_noise_ratio.\n"
            "- SSIM: NO esta en el paper de DIP. Metrica estandar de calidad de imagen "
            "(Wang et al., IEEE TIP 2004); la agregamos porque el PSNR solo no distingue "
            "si el error cae sobre la frontera de transicion. skimage.metrics."
            "structural_similarity.\n"
            "- MAE: NO esta en el paper de DIP. Estadistica basica (np.abs(orig-rec)."
            "mean()); la agregamos porque queda en unidades de SoC y es interpretable "
            "fisicamente."
        ),
    )

    # 5 -- codigo que genera las metricas
    add_code_slide(
        prs,
        "El codigo que calcula MAE / PSNR / SSIM  (restorationGRIS.py)",
        "from skimage.metrics import peak_signal_noise_ratio as compare_psnr\n"
        "from skimage.metrics import structural_similarity  as compare_ssim\n"
        "\n"
        "# --- durante el entrenamiento, cada 100 iteraciones ---\n"
        "psrn        = compare_psnr(img_np, out_np)                    # PSNR imagen completa\n"
        "psrn_masked = compare_psnr(img_masked, out_np * img_mask_np)  # PSNR solo observados\n"
        "ssim_full   = compare_ssim(img_np[0], out_clip[0], data_range=1.0)\n"
        "metrics_log.append((i, psrn, psrn_masked, ssim_full))\n"
        "\n"
        "# --- al terminar la corrida ---\n"
        "abs_err    = np.abs(img_np - out_np)      # mapa de error por pixel\n"
        "final_psnr = compare_psnr(img_np, out_np)\n"
        "final_ssim = compare_ssim(img_np[0], out_np[0], data_range=1.0)\n"
        "final_mae  = float(abs_err.mean())        # MAE = promedio del mapa de error",
        note="img_np = original (verdad de terreno)   ·   out_np = reconstruccion DIP   "
             "·   img_mask_np = mascara (1 = pixel visible).",
    )

    # 6 -- como leer cada figura
    add_full_image_slide(
        prs,
        "Como leer cada figura de resultado",
        os.path.join(R, "mf0.95", "comparison_annotated.png"),
        caption="Original | Enmascarada (lo que ve DIP) | Reconstruida | |error| "
                "(mapa de calor: negro = exacto, amarillo = error grande; la barra da el "
                "valor en SoC). El MAE del titulo es el promedio de ese mapa.",
        width=12.7, top=2.4,
    )

    # 7..13 -- las 7 corridas, una por porcentaje (128x128)
    for mf in MASKS:
        psnr, ssim, mae = DATA[128][mf]
        add_full_image_slide(
            prs,
            "Reconstruccion 128x128  -  %d %% de puntos ocultos" % (mf * 100),
            os.path.join(R, "mf%.2f" % mf, "comparison_annotated.png"),
            caption="%d puntos visibles de 16384    |    PSNR %.1f dB    |    SSIM %.4f    "
                    "|    MAE %.4f  (%.2f puntos porcentuales de SoC)"
                    % (visibles(mf, 128), psnr, ssim, mae, mae * 100),
            width=12.7, top=2.4,
        )

    # 14 -- tabla resumen
    add_table_slide(prs, "Resumen numerico  -  barrido 128x128", 128)

    # 15 -- figura resumen (curvas)
    add_full_image_slide(
        prs,
        "Fidelidad vs % de puntos ocultos",
        resumen_png,
        caption="Degradacion controlada. A igual % oculto, 128x128 aguanta mejor porque "
                "conserva mas puntos en valor absoluto.",
        width=12.9, top=2.4,
    )

    # 16 -- conclusion
    add_bullets_slide(
        prs,
        "Cuantos puntos hacen falta",
        [
            "Reconstruccion fiel con margen  (~5 % de los pixeles visibles):",
            "- ~200 puntos a 64x64,  ~800 a 128x128.  SSIM > 0.996, MAE < 0.15 pp.",
            "Minimo aceptable  (~2 %):  ~80 puntos a 64x64, ~330 a 128x128;",
            "- la frontera de transicion se distorsiona.",
            "Por debajo de ~1 %  (~40-160 puntos):  se rompe (MAE x3-x5, estructura espuria).",
            "",
            "El error se concentra sobre la diagonal de transicion (SoC de ~1 a ~0):",
            "- esa es la unica parte dificil de interpolar.",
        ],
    )

    out = os.path.join(OUT_DIR, PRES_NAME + ".pptx")
    prs.save(out)

    # snapshot del generador usado, para poder reproducir exactamente esta version
    shutil.copy(__file__, os.path.join(OUT_DIR, "make_" + PRES_NAME + ".py"))

    print("Presentacion guardada en:", out,
          "(%d diapositivas)" % len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
