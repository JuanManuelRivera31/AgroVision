"""Funciones de visualización (OpenCV para dibujar, Matplotlib para mostrar)."""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ReglasMadurez
from .procesamiento import a_rgb


def mostrar_grilla(imagenes: list, titulos: list, columnas: int = 3,
                   tam_celda: float = 4.5, cmap_gris: str = "gray", titulo: str = ""):
    """Muestra una lista de imágenes BGR o en escala de grises."""
    filas = int(np.ceil(len(imagenes) / columnas))
    fig, ejes = plt.subplots(filas, columnas,
                             figsize=(tam_celda * columnas, tam_celda * 0.8 * filas))
    ejes = np.atleast_1d(ejes).ravel()
    for ax, img, t in zip(ejes, imagenes, titulos):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap_gris)
        else:
            ax.imshow(a_rgb(img))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    for ax in ejes[len(imagenes):]:
        ax.axis("off")
    if titulo:
        fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def color_clase_mpl(rangos: list) -> tuple:
    """Color RGB (0-1) representativo de una clase de color: tono y brillo medios de su primer intervalo."""
    (h0, _, v0), (h1, _, v1) = rangos[0]
    hsv = np.uint8([[[(h0 + h1) // 2, 200, (v0 + v1) // 2]]])
    r, g, b = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return (r / 255, g / 255, b / 255)


def dibujar_detecciones(img: np.ndarray, detecciones: list, df: pd.DataFrame | None = None,
                        reglas: ReglasMadurez | None = None,
                        columna_cat: str = "categoria", grosor: int = 2,
                        rechazadas: list | None = None) -> np.ndarray:
    """Bounding boxes con número y color por categoría (cv2.rectangle/putText).

    Sin ``reglas`` (o sin categoría) las cajas se dibujan en cian.
    """
    colores = reglas.colores_bgr if reglas else {}
    salida = img.copy()
    if rechazadas:
        for r in rechazadas:
            cv2.drawContours(salida, [r.contorno], -1, (160, 160, 160), 1)
    cats = {}
    if df is not None and len(df) and columna_cat in df:
        cats = dict(zip(df["id"], df[columna_cat]))
    escala_txt = max(0.5, img.shape[1] / 1600)
    for d in detecciones:
        color = colores.get(cats.get(d.id), (255, 255, 0))
        x, y, w, h = d.bbox
        cv2.drawContours(salida, [d.contorno], -1, color, 1)
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, grosor)
        texto = f"{d.id}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala_txt, 2)
        cv2.rectangle(salida, (x, max(y - th - 6, 0)), (x + tw + 6, max(y, th + 6)), color, -1)
        cv2.putText(salida, texto, (x + 3, max(y - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    escala_txt, (255, 255, 255), 2, cv2.LINE_AA)
    return salida


def superponer_mascara(img: np.ndarray, mascara: np.ndarray,
                       color=(255, 0, 255), alfa: float = 0.45) -> np.ndarray:
    capa = img.copy()
    capa[mascara > 0] = color
    return cv2.addWeighted(capa, alfa, img, 1 - alfa, 0)


def mapa_disparidad_color(disp: np.ndarray) -> np.ndarray:
    """Disparidad normalizada (percentiles 2-98) con mapa de color; inválidos en negro."""
    valido = np.isfinite(disp)
    out = np.zeros((*disp.shape, 3), np.uint8)
    if not valido.any():
        return out
    lo, hi = np.percentile(disp[valido], [2, 98])
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    norm = (np.nan_to_num(norm) * 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[~valido] = 0
    return out


def lineas_epipolares(img_izq, img_der, cada: int = 40) -> np.ndarray:
    """Par lado a lado con líneas horizontales para inspeccionar la rectificación."""
    h = min(img_izq.shape[0], img_der.shape[0])
    par = np.hstack([img_izq[:h], img_der[:h]])
    for y in range(0, h, cada):
        cv2.line(par, (0, y), (par.shape[1], y), (0, 255, 0), 1)
    return par


def panel_resultados(img_original, img_procesada, img_detecciones, df: pd.DataFrame,
                     reglas: ReglasMadurez,
                     columna_tam: str = "diametro_equiv_px", etiqueta_tam: str = "Diámetro equivalente (px)",
                     titulo: str = "AgroVisión — Resultados"):
    """Panel final para la sustentación: imágenes, conteos y gráficos."""
    categorias, colores_mpl = reglas.categorias, reglas.colores_mpl
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    for i, (im, t) in enumerate([(img_original, "Imagen original"),
                                 (img_procesada, "Imagen procesada (máscara)"),
                                 (img_detecciones, "Detecciones y clasificación")]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im if im.ndim == 2 else a_rgb(im), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=12)
        ax.axis("off")

    conteos = (df["categoria"].value_counts().reindex(categorias, fill_value=0)
               if len(df) else pd.Series(0, index=categorias))
    colores = [colores_mpl[c] for c in categorias]

    ax = fig.add_subplot(gs[1, 0])
    barras = ax.bar(categorias, conteos.values, color=colores, edgecolor="black")
    ax.bar_label(barras, fontsize=12, fontweight="bold")
    ax.set_title(f"Conteo por madurez (total = {len(df)})", fontsize=12)
    ax.set_ylabel("Número de frutos")
    ax.set_ylim(0, max(conteos.max(), 1) * 1.25)

    ax = fig.add_subplot(gs[1, 1])
    if len(df) and columna_tam in df and df[columna_tam].notna().any():
        orden = df.sort_values("id")
        ax.bar(orden["id"].astype(str), orden[columna_tam].fillna(0),
               color=[colores_mpl[c] for c in orden["categoria"]], edgecolor="black")
        media = orden[columna_tam].mean()
        ax.axhline(media, color="black", ls="--", lw=1, label=f"media = {media:.1f}")
        ax.legend()
    ax.set_title("Tamaño por fruto", fontsize=12)
    ax.set_xlabel("ID del fruto")
    ax.set_ylabel(etiqueta_tam)

    ax = fig.add_subplot(gs[1, 2])
    cx, cy = reglas.ejes_grafico
    if len(df):
        ax.scatter(df[f"pct_{cx}"] * 100, df[f"pct_{cy}"] * 100,
                   c=[colores_mpl[c] for c in df["categoria"]],
                   s=80, edgecolors="black")
        for _, f in df.iterrows():
            ax.annotate(str(int(f["id"])), (f[f"pct_{cx}"] * 100, f[f"pct_{cy}"] * 100),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"% superficie {cx}")
    ax.set_ylabel(f"% superficie {cy}")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title("Espacio de decisión de madurez", fontsize=12)
    ax.grid(alpha=0.3)

    fig.suptitle(titulo, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig
