#!/usr/bin/env python
# coding: utf-8
"""Genera presentacion2_09_09.pptx: segunda entrega de la etapa DIP sobre el
diagrama de fases del modelo del continuo. Cubre las cinco cosas que quedaron de
la reunion anterior:

  1. puntos en la frontera de fase y sin puntos en ella
  2. codigo ordenado en un paquete
  3. barrido 98 / 99 / 99.5 / 99.7 / 99.8 / 99.9 (+ grilla fina) y valor estandar
  4. error en funcion de la iteracion -> cuando cortar
  5. varios g (los negativos son los que mas cuestan)

Lee las metricas de archive/ y arma las figuras nuevas en la propia carpeta.
Mismo estilo y estructura que presentacion1_02_09.

Uso:  ./venv/bin/python presentaciones/presentacion2_09_09/make_presentacion2_09_09.py
"""
import csv
import glob
import os
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "presentaciones", "presentacion2_09_09")
PRES_NAME = "presentacion2_09_09"

A_FRONT = os.path.join(REPO, "archive", "dip_frontier")
A_GMAP = os.path.join(REPO, "archive", "phase_diagram_g")
A_GSW = os.path.join(REPO, "archive", "dip_gsweep")
A_GFIX = os.path.join(REPO, "archive", "dip_gsweep_frontier")

MF_G = ["0.90", "0.95", "0.98", "0.99", "0.995"]
MF_GFIX = ["0.900", "0.950", "0.980", "0.990", "0.995"]


# ------------------------------------------------------------ lectura ----------
def final_row(path):
    """Devuelve (psnr, ssim, mae, fallbacks) de la fila final de un metrics.csv."""
    psnr = ssim = mae = fb = None
    if not os.path.isfile(path):
        return None
    for row in csv.reader(open(path)):
        if not row:
            continue
        if row[0] == "final":
            psnr, ssim = float(row[1]), float(row[3])
        elif row[0] == "mae":
            mae = float(row[1])
        elif row[0] == "fallbacks":
            fb = int(row[1])
    return psnr, ssim, mae, fb


def curve(path):
    it, pf, pm = [], [], []
    for r in csv.DictReader(open(path)):
        if r["iter"] in ("final", "mae", "fallbacks"):
            continue
        it.append(int(r["iter"]))
        pf.append(float(r["psnr_full"]))
        pm.append(float(r["psnr_masked"]) if r["psnr_masked"] else np.nan)
    return np.array(it), np.array(pf), np.array(pm)


GS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 2.0, 4.0]
GS_FIX = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]


def gsweep_matrix():
    P = np.full((len(GS), len(MF_G)), np.nan)
    S = np.full((len(GS), len(MF_G)), np.nan)
    for i, g in enumerate(GS):
        for j, m in enumerate(MF_G):
            r = final_row(os.path.join(A_GSW, "g%.1f" % g, "mf" + m, "metrics.csv"))
            if r:
                P[i, j], S[i, j] = r[0], r[1]
    return P, S


def band_width(g):
    soc = np.load(os.path.join(A_GMAP, "g%.1f" % g, "soc.npy"))
    return float(((soc > 0.05) & (soc < 0.95)).mean())


# ------------------------------------------------------- figuras nuevas --------
def fig_iter_curves(path):
    cases = [
        (os.path.join(A_GSW, "g0.0", "mf0.98", "metrics.csv"),
         "g=0,  2% de puntos  (facil)", "tab:green"),
        (os.path.join(A_FRONT, "mf0.980", "uniform", "metrics.csv"),
         "g=0,  2%  (exp. frontera)", "tab:blue"),
        (os.path.join(A_GSW, "g-4.0", "mf0.90", "metrics.csv"),
         "g=-4,  10% de puntos", "tab:orange"),
        (os.path.join(A_GSW, "g-4.0", "mf0.98", "metrics.csv"),
         "g=-4,  2% de puntos  (dificil)", "tab:red"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for p, lab, c in cases:
        if not os.path.isfile(p):
            continue
        it, pf, pm = curve(p)
        axs[0].plot(it, pf, "-", color=c, label=lab)
        axs[1].plot(it, pm, "-", color=c, label=lab)
        k = int(np.nanargmax(pf))
        axs[0].plot(it[k], pf[k], "o", color=c, ms=6)
    axs[0].set_title("PSNR imagen completa  (la verdad de terreno)")
    axs[0].set_ylabel("PSNR [dB]")
    axs[1].set_title("PSNR sobre pixeles observados  (lo unico que ve el algoritmo)")
    axs[1].set_ylabel("PSNR [dB]")
    for ax in axs:
        ax.set_xlabel("iteracion")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Error vs iteracion:  en los casos dificiles el error real toca fondo "
                 "temprano y despues sobreajusta", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura:", path)


def fig_gfix_delta(path):
    dP = np.full((len(GS_FIX), len(MF_GFIX)), np.nan)
    dS = np.full((len(GS_FIX), len(MF_GFIX)), np.nan)
    for i, g in enumerate(GS_FIX):
        for j, m in enumerate(MF_GFIX):
            fm = final_row(os.path.join(A_GFIX, "g%.1f" % g, "mf" + m,
                                        "frontier_mix", "metrics.csv"))
            un = final_row(os.path.join(A_GFIX, "g%.1f" % g, "mf" + m,
                                        "uniform", "metrics.csv"))
            if fm and un:
                dP[i, j] = fm[0] - un[0]
                dS[i, j] = fm[1] - un[1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, D, ttl, vlim in [
        (axs[0], dP, "Delta PSNR  [dB]", 22),
        (axs[1], dS, "Delta SSIM", 0.35)]:
        im = ax.imshow(D, aspect="auto", cmap="RdBu", vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(len(MF_GFIX)))
        ax.set_xticklabels(["%g%%" % (100 * float(m)) for m in MF_GFIX])
        ax.set_yticks(range(len(GS_FIX)))
        ax.set_yticklabels(["%+.1f" % g for g in GS_FIX])
        ax.set_xlabel("% de puntos ocultos")
        ax.set_ylabel("g")
        ax.set_title(ttl)
        for i in range(len(GS_FIX)):
            for j in range(len(MF_GFIX)):
                v = D[i, j]
                if not np.isnan(v):
                    ax.text(j, i, "%+.1f" % v if "PSNR" in ttl else "%+.2f" % v,
                            ha="center", va="center", fontsize=8,
                            color="black" if abs(im.norm(v) - 0.5) < 0.28 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mascara pesada en la frontera (frontier_mix) menos uniforme  -  "
                 "azul = mejora, rojo = empeora", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura:", path)


def fig_gmaps_montage(path):
    """Los 11 mapas del continuo por g, en 2 filas x 6, apaisado y con colorbar
    unica (reemplaza archive/phase_diagram_g/montage_gmaps.png, que salia en
    formato vertical y con un panel vacio)."""
    fig, axs = plt.subplots(2, 6, figsize=(15.5, 6.0))
    fig.subplots_adjust(hspace=0.45)
    axs = axs.ravel()
    im = None
    for k, g in enumerate(GS):
        ax = axs[k]
        soc = np.load(os.path.join(A_GMAP, "g%.1f" % g, "soc.npy"))
        im = ax.imshow(soc, origin="lower", extent=[-4, 2, -4, 2],
                       vmin=0, vmax=1, cmap="viridis")
        ax.set_title("g = %g   (banda %.0f%%)" % (g, 100 * band_width(g)), fontsize=10)
        ax.set_xticks([-4, -2, 0, 2])
        ax.set_yticks([-4, -2, 0, 2])
        ax.tick_params(labelsize=8)
        if k % 6 == 0:
            ax.set_ylabel("log Xi", fontsize=9)
        if k >= 6:
            ax.set_xlabel("log l", fontsize=9)
    for j in range(len(GS), len(axs)):
        axs[j].axis("off")
    fig.suptitle("Mapas del continuo  SoC_max(log Xi, log l)  por g (Frumkin)  -  128x128",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95], h_pad=2.5)
    cax = fig.add_axes([0.945, 0.12, 0.011, 0.76])
    fig.colorbar(im, cax=cax, label="SoC_max  (violeta = 0,  amarillo = 1)")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura:", path)


def fig_frontier_grid(path):
    """Reconstrucciones por distribucion de puntos, 4 columnas x 2 filas (dos
    regimenes de N), apaisado (reemplaza archive/dip_frontier/summary_grid.png,
    que era 6x4 y quedaba mas alto que la slide)."""
    dists = ["uniform", "frontier", "frontier_mix", "spread"]
    rows = [("0.980", "2% obs  (N=328)"), ("0.997", "0.3% obs  (N=49)")]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.6))
    for i, (mf, rlab) in enumerate(rows):
        for j, d in enumerate(dists):
            ax = axs[i][j]
            arr = np.squeeze(np.load(os.path.join(A_FRONT, "mf" + mf, d, "restored.npy")))
            if arr.ndim == 3:
                arr = arr[0]
            arr = arr[1:-1, 1:-1]  # sacar el ReflectionPad2d(1)
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            r = final_row(os.path.join(A_FRONT, "mf" + mf, d, "metrics.csv"))
            if r:
                ax.set_xlabel("PSNR %.1f   SSIM %.3f" % (r[0], r[1]), fontsize=9)
            if i == 0:
                ax.set_title(d, fontsize=12)
            if j == 0:
                ax.set_ylabel(rlab, fontsize=11)
    fig.suptitle("Reconstruccion DIP por distribucion de puntos  -  mismo N por fila",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura:", path)


FR_DISTS = ["uniform", "frontier", "frontier_mix", "spread"]
FR_DETAIL = [
    ("0.980", "N = 328 puntos  (2% observado)"),
    ("0.990", "N = 164 puntos  (1% observado)"),
    ("0.995", "N = 82 puntos  (0.5% observado)"),
    ("0.997", "N = 49 puntos  (0.3% observado)"),
    ("0.998", "N = 33 puntos  (0.2% observado)"),
    ("0.999", "N = 16 puntos  (0.1% observado)"),
]


A_MASKS = os.path.join(REPO, "data", "restoration", "masks")


def _frontier_obs_xy(mf, d):
    """coords (x, y) de los pixeles observados, del .npy booleano exacto
    (data/restoration/masks/mf<mf>/mask_<d>.npy, 128x128)."""
    m = np.load(os.path.join(A_MASKS, "mf" + mf, "mask_%s.npy" % d))
    ys, xs = np.where(m)
    return xs, ys


def fig_frontier_detail(path, mf, title):
    """Una figura por regimen de N: fila de arriba = donde caen los puntos
    observados (rojo, sobre el mapa real); fila de abajo = la reconstruccion
    DIP de cada una de las 4 distribuciones."""
    orig = np.squeeze(
        np.load(os.path.join(A_FRONT, "mf" + mf, FR_DISTS[0], "original.npy")))[1:-1, 1:-1]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.8))
    for j, d in enumerate(FR_DISTS):
        xs, ys = _frontier_obs_xy(mf, d)
        a0 = axs[0][j]
        a0.imshow(orig, cmap="gray", vmin=0, vmax=1, alpha=0.35)
        a0.scatter(xs, ys, s=10, c="#d62728", edgecolors="none")
        a0.set_xlim(0, orig.shape[1])
        a0.set_ylim(orig.shape[0], 0)
        a0.set_xticks([])
        a0.set_yticks([])
        a0.set_title(d, fontsize=12)
        a0.set_xlabel("%d puntos" % len(xs), fontsize=9)

        rec = np.squeeze(
            np.load(os.path.join(A_FRONT, "mf" + mf, d, "restored.npy")))[1:-1, 1:-1]
        a1 = axs[1][j]
        a1.imshow(rec, cmap="gray", vmin=0, vmax=1)
        a1.set_xticks([])
        a1.set_yticks([])
        r = final_row(os.path.join(A_FRONT, "mf" + mf, d, "metrics.csv"))
        if r:
            a1.set_xlabel("PSNR %.1f   SSIM %.3f" % (r[0], r[1]), fontsize=9)
    axs[0][0].set_ylabel("puntos observados\n(sobre el mapa real)", fontsize=10)
    axs[1][0].set_ylabel("reconstruccion DIP", fontsize=10)
    fig.suptitle("Distribucion de puntos vs reconstruccion  -  " + title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura:", path)


# --------------------------------------------------- frontera: como se define --
SOC128 = os.path.join(REPO, "archive", "phase_diagram", "phase_diagram_soc_128.npy")


def _edge_field(soc, sigma=1.0):
    """|grad SoC| suavizado y normalizado a [0,1]  (misma receta que
    dip/frontier_mask.py: gaussiana sigma=1, hypot, gaussiana, /max).
    1 = frontera de fase, 0 = meseta."""
    gy, gx = np.gradient(ndi.gaussian_filter(soc.astype(float), sigma))
    gmag = ndi.gaussian_filter(np.hypot(gx, gy), sigma)
    return gmag / max(gmag.max(), 1e-12)


def fig_frontier_method(path):
    """Ilustra el pipeline: mapa SoC -> |grad| suavizado -> peso ^gamma."""
    soc = np.flipud(np.load(SOC128))  # a la orientacion del PNG (fila 0 = xi alto)
    edge = _edge_field(soc)
    lvl = np.quantile(edge, 0.90)
    w_edge = (edge ** 3 + 1e-4)
    w_edge = w_edge / w_edge.max()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.7))
    axs[0].imshow(soc, cmap="viridis", vmin=0, vmax=1)
    axs[0].set_title("mapa SoC_max  (lo que ve DIP)", fontsize=11)
    im1 = axs[1].imshow(edge, cmap="magma", vmin=0, vmax=1)
    axs[1].contour(edge, levels=[lvl], colors="cyan", linewidths=1.2)
    axs[1].set_title(r"campo de frontera  $|\nabla \mathrm{SoC}|$" + "\nsuavizado y normalizado",
                     fontsize=11)
    axs[2].imshow(w_edge ** 0.35, cmap="magma", vmin=0, vmax=1)
    axs[2].set_title(r"peso de sorteo 'frontier'  $\propto \mathrm{edge}^{3}$", fontsize=11)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im1, ax=axs, fraction=0.025, pad=0.02)
    fig.suptitle("Como se distingue la frontera:  gradiente del mapa -> suavizar -> "
                 "elevar a una potencia (cian = curva de nivel del percentil 90)", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("figura:", path)


GS_SHOW = [-4.0, -3.0, -2.0, -1.0, 0.0, 2.0, 4.0]
MF_G_SHOW = [("0.90", "10% observado"), ("0.98", "2% observado"),
             ("0.995", "0.5% observado")]


def fig_gsweep_detail(path, mf, pct_label):
    """Fila de arriba: mapa real de cada g con la frontera marcada (linea blanca,
    igual criterio que el experimento de frontera).  Fila de abajo: la
    reconstruccion DIP a esa fraccion (mascara uniforme)."""
    fig, axs = plt.subplots(2, len(GS_SHOW), figsize=(15.5, 5.1))
    for j, g in enumerate(GS_SHOW):
        d = os.path.join(A_GSW, "g%.1f" % g, "mf" + mf)
        orig = np.squeeze(np.load(os.path.join(d, "original.npy")))[1:-1, 1:-1]
        rec = np.squeeze(np.load(os.path.join(d, "restored.npy")))[1:-1, 1:-1]
        edge = _edge_field(orig)
        lvl = np.quantile(edge, 0.90)
        a0, a1 = axs[0][j], axs[1][j]
        a0.imshow(orig, cmap="viridis", vmin=0, vmax=1)
        a0.contour(edge, levels=[lvl], colors="white", linewidths=1.0)
        a0.set_title("g = %g" % g, fontsize=11)
        a1.imshow(rec, cmap="viridis", vmin=0, vmax=1)
        a1.contour(edge, levels=[lvl], colors="white", linewidths=1.0)
        r = final_row(os.path.join(d, "metrics.csv"))
        if r:
            a1.set_xlabel("PSNR %.1f\nSSIM %.3f" % (r[0], r[1]), fontsize=9)
        for a in (a0, a1):
            a.set_xticks([])
            a.set_yticks([])
    axs[0][0].set_ylabel("mapa real\n(linea = frontera)", fontsize=10)
    axs[1][0].set_ylabel("reconstruccion DIP", fontsize=10)
    fig.suptitle("Reconstruccion segun g  -  mascara uniforme,  " + pct_label, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figura:", path)


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
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE
    tb2 = s.shapes.add_textbox(Inches(0.7), Inches(3.7), Inches(12), Inches(1.5))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(19)
    p2.font.color.rgb = GREY
    return s


def add_bullets_slide(prs, title, bullets, img=None, img_left=7.3, img_top=1.6,
                      img_width=5.7, note=None, body_size=18):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, 27)
    body_w = 12 if img is None else (img_left - 0.9)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(body_w), Inches(5.4))
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
        s.shapes.add_picture(img, Inches(img_left), Inches(img_top), width=Inches(img_width))
    if note:
        nb = s.shapes.add_textbox(Inches(0.6), Inches(6.5), Inches(12.2), Inches(0.9))
        tf2 = nb.text_frame
        tf2.word_wrap = True
        for j, line in enumerate(note.split("\n")):
            para = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
            para.text = line
            para.font.size = Pt(10)
            para.font.italic = True
            para.font.color.rgb = GREY
    return s


SLIDE_W, SLIDE_H = 13.33, 7.5


def _fit(img, max_w, max_h):
    """Ancho/alto en pulgadas para que <img> entre en la caja max_w x max_h
    conservando su relacion de aspecto (nunca se agranda mas alla de max_w)."""
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
    band_top = 1.5
    band_bot = 6.75 if caption else 7.25
    max_w = float(width) if width else 12.6
    max_h = band_bot - band_top
    w, h = _fit(img, max_w, max_h)
    left = (SLIDE_W - w) / 2
    t = band_top + (max_h - h) / 2
    s.shapes.add_picture(img, Inches(left), Inches(t), width=Inches(w), height=Inches(h))
    if caption:
        cb = s.shapes.add_textbox(Inches(0.6), Inches(band_bot + 0.05),
                                  Inches(12.1), Inches(0.6))
        cp = cb.text_frame.paragraphs[0]
        cp.text = caption
        cp.font.size = Pt(12)
        cp.font.italic = True
        cp.font.color.rgb = GREY
        cp.alignment = PP_ALIGN.CENTER
    return s


def add_two_image_slide(prs, title, img_l, img_r, cap_l, cap_r, caption=None, width=6.2):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    for img, cx, cap in [(img_l, SLIDE_W * 0.27, cap_l), (img_r, SLIDE_W * 0.75, cap_r)]:
        w, h = _fit(img, width, 4.2)
        t = 2.15 + (4.2 - h) / 2
        s.shapes.add_picture(img, Inches(cx - w / 2), Inches(t),
                             width=Inches(w), height=Inches(h))
        tb = s.shapes.add_textbox(Inches(cx - width / 2), Inches(1.55),
                                  Inches(width), Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = cap
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = BLUE
        p.alignment = PP_ALIGN.CENTER
    if caption:
        cb = s.shapes.add_textbox(Inches(0.6), Inches(6.7), Inches(12.1), Inches(0.7))
        cp = cb.text_frame.paragraphs[0]
        cp.text = caption
        cp.font.size = Pt(12)
        cp.font.italic = True
        cp.font.color.rgb = GREY
        cp.alignment = PP_ALIGN.CENTER
    return s


def add_table_slide(prs, title, headers, rows, note=None, left=1.0, col_w=None,
                    font=12):
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
    iter_png = os.path.join(OUT_DIR, "iter_curves.png")
    gfix_png = os.path.join(OUT_DIR, "gfix_delta.png")
    gmaps_png = os.path.join(OUT_DIR, "gmaps_montage.png")
    frgrid_png = os.path.join(OUT_DIR, "frontier_grid.png")
    frmethod_png = os.path.join(OUT_DIR, "frontier_method.png")
    fig_iter_curves(iter_png)
    fig_gfix_delta(gfix_png)
    fig_gmaps_montage(gmaps_png)
    fig_frontier_grid(frgrid_png)
    fig_frontier_method(frmethod_png)

    P, S = gsweep_matrix()
    band = {g: band_width(g) for g in GS}

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # 1 -- portada
    add_title_slide(
        prs,
        "Reconstruccion del diagrama de fases con Deep Image Prior",
        "Interpolacion de mapas SoC_max desde muestreo disperso  ·  segunda entrega  ·  "
        "Santiago Afonso  ·  09/09/2026",
    )

    # 2 -- lo que quedo para esta entrega
    add_bullets_slide(
        prs,
        "Que quedo para esta entrega",
        [
            "En la reunion anterior quedaron cinco cosas para mirar:",
            "1.  Mascaras con puntos sobre la frontera de fase y mascaras sin puntos en ella.",
            "2.  Ordenar el codigo, que venia como varios scripts sueltos.",
            "3.  Barrer el porcentaje de puntos ocultos hasta muy poco: 98, 99, 99.5, 99.7, 99.8 y 99.9 %.",
            "4.  El error en funcion de la iteracion, para decidir en que momento conviene cortar.",
            "5.  Repetir todo para varios g; los negativos son los que mas cuestan.",
            "",
            "Todo sobre el mapa del modelo del continuo (galpynostatic), 128x128, con mascara "
            "Bernoulli salvo cuando se aclara otra cosa.",
        ],
        body_size=17,
    )

    # 3 -- como leer cada figura
    add_full_image_slide(
        prs,
        "Como leer cada figura de resultado",
        os.path.join(A_FRONT, "mf0.980", "uniform", "comparison_annotated.png"),
        caption="Original | Enmascarada (lo que ve DIP: solo los puntos observados) | Reconstruida | "
                "|error| (negro = exacto, amarillo = error grande; la barra da el valor en SoC). "
                "DIP ajusta la red para que pase por los puntos observados y el resto lo completa el prior.",
        width=12.8, top=2.5,
    )

    # 4 -- el codigo, ordenado
    add_bullets_slide(
        prs,
        "2 · El codigo, ordenado",
        [
            "Antes: tres scripts casi iguales sueltos en la raiz (restoration, restorationGRIS, "
            "restorationRGB), once .slurm con rutas fijas al home, y notebooks y salidas mezclados.",
            "",
            "Ahora esta todo en un paquete:",
            "- dip/  -  el codigo propio: un unico runner restoration parametrizado, mas metrics, "
            "phase_diagram, frontier_mask y sample_points.  Se corre como  python -m dip.<modulo>.",
            "- slurm/  -  todos los jobs juntos, sin rutas al home.",
            "- archive/  -  las salidas viejas;  results/  -  las nuevas.",
            "- El runner toma todo por variables de entorno (IMAGE_PATH, MASK_FRAC o MASK_PATH, "
            "NUM_ITER, SEED, REG_NOISE_STD, ...), asi que cada barrido es un solo script.",
        ],
        body_size=16,
    )

    # 4b -- como se define la frontera
    add_bullets_slide(
        prs,
        "1 · Como se define la frontera",
        [
            "La 'frontera' es el modulo del gradiente del mapa, |grad SoC|, calculado sobre la grilla "
            "cruda, suavizado con una gaussiana (sigma = 1 px) y normalizado a [0, 1]:  vale 1 sobre "
            "la transicion y 0 en las mesetas.  La linea de las figuras es la curva de nivel del "
            "percentil 90.",
            "",
            "Con eso armo cuatro familias de mascaras, todas con la misma cantidad de puntos para "
            "cada fraccion:",
            "- uniform  -  Bernoulli, todos los puntos con el mismo peso; es el baseline.",
            "- frontier  -  peso proporcional a edge^3: casi todos los puntos caen sobre la transicion.",
            "- frontier_mix  -  0.85 frontier + 0.15 uniforme: la mayoria en la frontera, algo repartido.",
            "- spread  -  peso proporcional a (1 - edge)^3: los puntos evitan la frontera, van a las mesetas.",
            "",
            "Las mascaras estan anidadas: se sortea una vez el N mas grande de cada familia y las "
            "fracciones chicas son un prefijo (los puntos de mf0.999 estan dentro de mf0.998, etc.).  "
            "Semilla fija, 42.",
        ],
        body_size=15,
    )
    add_full_image_slide(
        prs,
        "1 · El campo de frontera, paso a paso",
        frmethod_png,
        caption="Izquierda: el mapa.  Centro: |grad SoC| suavizado y normalizado, con la curva del "
                "percentil 90 en cian (eso es 'la frontera').  Derecha: el peso de sorteo de la "
                "familia frontier, proporcional a edge^3, concentrado sobre esa curva.",
    )

    # 5 -- experimento frontera: la grilla
    add_full_image_slide(
        prs,
        "1 · Puntos sobre la frontera y puntos repartidos  -  misma cantidad, distinta ubicacion",
        frgrid_png,
        caption="Columnas: uniform (Bernoulli), frontier (todo sobre |grad SoC|), frontier_mix "
                "(~85 % frontera + 15 % repartido) y spread (evita la frontera).  Arriba con 2 % de "
                "puntos observados, abajo con 0.3 %.  El detalle N por N esta en las slides siguientes.",
    )

    # 5b -- detalle: una slide por regimen de N, puntos observados + reconstruccion
    for mf, lab in FR_DETAIL:
        det_png = os.path.join(OUT_DIR, "frontier_detail_mf%s.png" % mf)
        fig_frontier_detail(det_png, mf, lab)
        add_full_image_slide(
            prs,
            "1 · Puntos observados y reconstruccion  -  " + lab,
            det_png,
            caption="Arriba, donde caen los puntos observados (en rojo) sobre el mapa real.  Abajo, "
                    "lo que reconstruye DIP a partir de solo esos puntos.  Misma cantidad en las "
                    "cuatro columnas, cambia la distribucion.",
        )

    # 6 -- frontera: metricas + tabla
    fr = {n: final_row(os.path.join(A_FRONT, "mf0.980", n, "metrics.csv"))
          for n in ["uniform", "spread", "frontier_mix", "frontier"]}
    add_table_slide(
        prs,
        "1 · Concentrar los puntos en la frontera es lo peor que se puede hacer",
        ["distribucion", "PSNR [dB]", "SSIM", "lectura  (con 2 % de puntos observados, N = 328)"],
        [
            ["uniform", "%.1f" % fr["uniform"][0], "%.3f" % fr["uniform"][1],
             "la mejor"],
            ["spread (evita la frontera)", "%.1f" % fr["spread"][0], "%.3f" % fr["spread"][1],
             "casi igual; un poco mejor cuando hay muy pocos puntos"],
            ["frontier_mix", "%.1f" % fr["frontier_mix"][0], "%.3f" % fr["frontier_mix"][1],
             "se cae rapido en cuanto bajan los puntos"],
            ["frontier (todo en la frontera)", "%.1f" % fr["frontier"][0],
             "%.3f" % fr["frontier"][1], "se queda plano en ~9 dB para cualquier N"],
        ],
        note="La razon: DIP no puede fijar las dos mesetas (SoC ~ 1 y SoC ~ 0, que son el ~80 % del "
             "area) si no tiene puntos ahi.\nCon los puntos repartidos fija esos niveles y el prior "
             "interpola solo la frontera suave.  Cubrir las mesetas pesa mas que apuntar a la frontera.",
        col_w=[3.0, 1.6, 1.3, 5.4], font=13,
    )

    # 7 -- cuantos puntos: la curva
    add_full_image_slide(
        prs,
        "3 · Cuantos puntos hacen falta  -  barrido de mf 0.80 a 0.999 (104 corridas)",
        os.path.join(A_FRONT, "plateau_vs_N.png"),
        caption="Calidad en funcion de N, una linea por distribucion.  El punteado marca el umbral "
                "SSIM 0.97 / PSNR 30.  Por debajo de ~0.5 % de puntos observados se derrumban todas.",
        width=12.9, top=2.4,
    )

    # 8 -- cuantos puntos: un valor estandar
    add_table_slide(
        prs,
        "3 · Un valor estandar para dejar de barrer N",
        ["criterio (distribucion uniform)", "N minimo", "% observado", "mf"],
        [
            ["reconstruccion confiable  (SSIM >= 0.97 y PSNR >= 30)", "~164", "1 %", "0.99"],
            ["casi perfecta  (SSIM >= 0.99)", "~492", "3 %", "0.97"],
            ["codo de la curva  (sumar puntos ya no aporta)", "~819", "5 %", "0.95"],
        ],
        note="Propuesta: fijar mf = 0.98 (2 % observado, N ~ 328), que da PSNR 34 / SSIM 0.98 con "
             "margen, y no volver a barrer N en los experimentos que siguen.\n"
             "De lo que quedo pedido (98/99/99.5/99.7/99.8/99.9): con 99.5 % (0.5 % observado) ya se "
             "rompe, y 99.7/99.8/99.9 estan todos en el regimen roto.",
        col_w=[6.0, 1.6, 2.0, 1.3], font=13,
    )

    # 9 -- error vs iteracion
    add_full_image_slide(
        prs,
        "4 · El error en funcion de la iteracion",
        iter_png,
        caption="El circulo marca el pico del PSNR real.  En 'g=-4, 2%' el PSNR real toca fondo cerca "
                "de la iteracion 300 y despues empeora, mientras el PSNR sobre los puntos observados "
                "sigue subiendo.",
        width=12.9, top=2.4,
    )

    # 10 -- cuando cortar
    add_bullets_slide(
        prs,
        "4 · En que momento conviene cortar",
        [
            "Casos faciles (g >= 0, muchos puntos): el PSNR real sube casi sin bajar hasta el final, "
            "asi que conviene dejar correr las 11000 iteraciones.",
            "",
            "Casos dificiles (pocos puntos, frente abrupto, g negativo): el PSNR real toca fondo "
            "temprano -de unos cientos a pocos miles de iteraciones- y despues la red se sobreajusta "
            "al ruido.",
            "",
            "El problema es que el PSNR sobre los puntos observados -lo unico que ve el algoritmo- "
            "sigue subiendo mientras el error real crece, asi que por si solo no sirve como senal de "
            "corte.  El backtracking por caida de PSNR ayuda, pero patina (10 a 18 rollbacks) en los "
            "peores casos.",
            "",
            "Conclusion: hace falta una senal de validacion, dejando afuera un subconjunto de puntos KMC.",
        ],
        body_size=16,
    )

    # 11 -- varios g: los mapas
    add_full_image_slide(
        prs,
        "5 · Varios g  -  la interaccion cambia el ancho de la transicion",
        gmaps_png,
        caption="Isoterma de Frumkin.  g < 0 (atractivo): transicion abrupta y localizada.  "
                "g > 0 (repulsivo): rampa ancha.  Las mesetas y el SoC medio (~0.43) son iguales para todo g.",
    )

    # 12 -- varios g: reconstruccion con mascara uniforme, heatmaps
    add_full_image_slide(
        prs,
        "5 · Reconstruccion con mascara uniforme sobre los 11 mapas  (55 corridas)",
        os.path.join(A_GSW, "gsweep_heatmaps.png"),
        caption="Cuanto mas negativo es g, peor reconstruye, para cualquier fraccion.  El error se "
                "concentra en la banda de transicion, que para g negativo es fina.",
        width=13.0, top=2.5,
    )

    # 12b -- varios g: reconstrucciones como imagenes, una slide por fraccion
    for mf, lab in MF_G_SHOW:
        gdet_png = os.path.join(OUT_DIR, "gsweep_detail_mf%s.png" % mf)
        fig_gsweep_detail(gdet_png, mf, lab)
        add_full_image_slide(
            prs,
            "5 · Reconstruccion segun g  -  " + lab,
            gdet_png,
            caption="Arriba, el mapa real de cada g con la frontera marcada (linea blanca, percentil "
                    "90 de |grad SoC|).  Abajo, la reconstruccion DIP con mascara uniforme.  Para g "
                    "negativo el frente fino se lava.",
        )

    # 13 -- varios g: tabla minimo funcional
    def cell(i, j):
        return "%.1f / %.2f" % (P[i, j], S[i, j])
    rows = []
    for i, g in enumerate(GS):
        oks = [MF_G[j] for j in range(len(MF_G)) if S[i, j] >= 0.97 and P[i, j] >= 30]
        best = max(oks, key=float) if oks else "-"
        minf = ("mf %s" % best) if best != "-" else "ninguno"
        rows.append(["%+.1f" % g, "%.1f%%" % (100 * band[g]),
                     cell(i, 2), cell(i, 3), minf])
    add_table_slide(
        prs,
        "5 · Minimo de puntos para una reconstruccion confiable, por g",
        ["g", "banda de frontera", "PSNR/SSIM con 2 %", "PSNR/SSIM con 1 %", "minimo que funciona"],
        rows,
        note="'minimo que funciona' = el mf mas alto (la menor cantidad de puntos) que todavia da "
             "SSIM >= 0.97 y PSNR >= 30.  g = -4 y -3.5 no llegan ni con 10 % de puntos.",
        col_w=[1.0, 2.0, 2.6, 2.6, 2.8], font=12,
    )

    # 14 -- un intento de arreglo para g negativo
    add_full_image_slide(
        prs,
        "5 · Un intento de arreglo para g negativo: cargar la mascara en la frontera",
        gfix_png,
        caption="frontier_mix por g (MIX_FRAC 0.40) y backtracking mas robusto (PSNR_DROP_TOL -8, "
                "NUM_ITER 16000, REG_NOISE_STD 0.01), contra la uniforme corrida en las mismas "
                "condiciones.  Azul = mejora.",
        width=12.9, top=2.4,
    )

    # 15 -- arreglo: ejemplo visual g=-4 mf0.90
    add_two_image_slide(
        prs,
        "5 · Ejemplo:  g = -4,  con 10 % de puntos  (mf 0.90)",
        os.path.join(A_GFIX, "g-4.0", "mf0.900", "uniform", "comparison_annotated.png"),
        os.path.join(A_GFIX, "g-4.0", "mf0.900", "frontier_mix", "comparison_annotated.png"),
        "uniforme   (PSNR 28.0 / SSIM 0.98)",
        "frontier_mix   (PSNR 48.5 / SSIM 0.99)",
        caption="+20 dB con la misma cantidad de puntos, solo redistribuidos hacia la frontera.  "
                "El escalon deja de estar lavado.",
        width=6.2,
    )

    # 16 -- arreglo: cuando ayuda y cuando no
    add_table_slide(
        prs,
        "5 · El arreglo ayuda, pero solo si hay puntos de sobra",
        ["regimen", "efecto de frontier_mix frente a uniforme"],
        [
            ["mf 0.90 - 0.95  (5-10 % observado)",
             "gran mejora: g=-4 mf0.90 pasa de PSNR 28 a 48 (+20 dB); g=-3.5, +24 dB.  Gana en casi todo g <= 0."],
            ["mf 0.98 - 0.995  (<= 2 % observado)",
             "empeora, a veces bastante (SSIM -0.1 a -0.6): al cargar la frontera, las mesetas quedan sin fijar."],
        ],
        note="Mejora el SSIM en 7 de 45 celdas -justo las de g negativo con >= 5 % de puntos, que "
             "eran el problema- y lo empeora en 32 (las de pocos puntos).\n"
             "Hay un piso fisico: con ~1 % de puntos, un frente fino mas dos mesetas no se fijan del "
             "todo, se repartan como se repartan.",
        col_w=[3.4, 8.0], font=13,
    )

    # 17 -- conclusiones
    add_bullets_slide(
        prs,
        "Conclusiones",
        [
            "DIP reconstruye el diagrama de fases desde muestreo ralo: con ~2 % de puntos uniformes "
            "(mf 0.98) alcanza SSIM ~0.98 para g >= -3.",
            "",
            "Donde poner los puntos: lo que importa es cubrir las mesetas; concentrarlos solo en la "
            "frontera es lo peor.",
            "",
            "El regimen dificil es g muy negativo, con frente abrupto: ahi el muestreo uniforme no "
            "alcanza, y hay que cargar la mascara en la frontera y ademas tener suficientes puntos (>= 5 %).",
            "",
            "Cuando cortar depende de la dificultad: en los casos dificiles hay sobreajuste temprano "
            "y el criterio interno (PSNR sobre los observados) no lo detecta.",
        ],
        body_size=16,
    )

    # 18 -- lo que falta y lo que sigue
    add_bullets_slide(
        prs,
        "Lo que falta y lo que sigue",
        [
            "Pendiente:",
            "- una sola imagen por g y una sola semilla.",
            "- las metricas son de la imagen completa, asi que las dominan las mesetas; falta un "
            "error medido solo sobre la banda de frontera para juzgar la transicion en si.",
            "- g <= -4 esta en el limite critico de la isoterma de Frumkin (transicion de primer orden).",
            "",
            "Proximos pasos:",
            "- una metrica de error restringida a la banda de frontera.",
            "- una senal de validacion (puntos KMC dejados afuera) para el corte por iteracion.",
            "- pasar de los mapas del continuo a puntos KMC reales.",
        ],
        body_size=16,
    )

    out = os.path.join(OUT_DIR, PRES_NAME + ".pptx")
    prs.save(out)
    snap = os.path.join(OUT_DIR, "make_" + PRES_NAME + ".py")
    if os.path.abspath(__file__) != os.path.abspath(snap):
        shutil.copy(__file__, snap)
    print("Presentacion guardada en:", out, "(%d diapositivas)" % len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
