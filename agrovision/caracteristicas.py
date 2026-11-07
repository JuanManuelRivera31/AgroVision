"""Extracción de características por fruto (siempre dentro de su máscara)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .config import ReglasMadurez
from .segmentacion import ResultadoSegmentacion, metricas_contorno


def porcentajes_color(hsv_px: np.ndarray, r: ReglasMadurez) -> dict:
    """Asigna cada píxel del fruto a una clase de color y devuelve los porcentajes.

    ``hsv_px``: arreglo (N, 3) con los píxeles HSV dentro de la máscara.
    Se descartan píxeles con S o V bajos (reflejos blancos, sombras profundas)
    porque su tono no es fiable. Cada píxel cuenta solo para la primera clase
    (en el orden del perfil) cuyo intervalo lo contiene.
    """
    px = hsv_px[(hsv_px[:, 1] >= r.s_min_valido) & (hsv_px[:, 2] >= r.v_min_valido)]
    n = len(px)
    if n == 0:
        return {**{c: 0.0 for c in r.columnas_pct}, "pct_otro": 1.0, "px_validos": 0}
    libre = np.ones(n, bool)
    salida = {}
    for nombre, rangos in r.colores.items():
        dentro = np.zeros(n, bool)
        for bajo, alto in rangos:
            dentro |= np.all((px >= bajo) & (px <= alto), axis=1)
        dentro &= libre
        libre &= ~dentro
        salida[f"pct_{nombre}"] = dentro.sum() / n
    return {**salida, "pct_otro": libre.sum() / n, "px_validos": n}


def histograma_h(hsv_px: np.ndarray, bins: int = 18) -> np.ndarray:
    """Histograma normalizado del tono (0-179) dentro del fruto."""
    hist, _ = np.histogram(hsv_px[:, 0], bins=bins, range=(0, 180))
    return hist / max(hist.sum(), 1)


def extraer_caracteristicas(res: ResultadoSegmentacion, reglas: ReglasMadurez) -> pd.DataFrame:
    """Tabla (una fila por fruto) con descriptores geométricos y de color.

    Las medidas en píxeles corresponden a la imagen procesada; las columnas
    ``*_px_orig`` están convertidas a la resolución original de la foto
    (dividiendo por la escala de redimensionamiento).
    """
    hsv = res.etapas["hsv"]
    lab = res.etapas["lab"]
    filas = []
    for d in res.detecciones:
        m = metricas_contorno(d.contorno, con_elipse=True)
        x, y, w, h = d.bbox
        sel = d.mascara_local > 0
        hsv_px = hsv[y:y + h, x:x + w][sel]
        lab_px = lab[y:y + h, x:x + w][sel]

        mom = cv2.moments(d.contorno)
        cx = mom["m10"] / mom["m00"] if mom["m00"] else x + w / 2
        cy = mom["m01"] / mom["m00"] if mom["m00"] else y + h / 2
        diam_eq = np.sqrt(4 * m["area"] / np.pi)

        fila = {
            "id": d.id,
            "area_px": m["area"],
            "perimetro_px": m["perimetro"],
            "circularidad": m["circularidad"],
            "solidez": m["solidez"],
            "ajuste_elipse": m["ajuste_elipse"],
            "relacion_aspecto": w / h,
            "bbox_x": x, "bbox_y": y,
            "ancho_bbox_px": w, "alto_bbox_px": h,
            "centroide_x": cx, "centroide_y": cy,
            "diametro_equiv_px": diam_eq,
            "diametro_equiv_px_orig": diam_eq / res.escala,
            "ancho_bbox_px_orig": w / res.escala,
            "alto_bbox_px_orig": h / res.escala,
            "H_media": float(hsv_px[:, 0].mean()),
            "H_std": float(hsv_px[:, 0].std()),
            "S_media": float(hsv_px[:, 1].mean()),
            "V_media": float(hsv_px[:, 2].mean()),
            # LAB de OpenCV: a* y b* desplazados +128 (128 = neutro).
            "a_media": float(lab_px[:, 1].mean()) - 128,
            "b_media": float(lab_px[:, 2].mean()) - 128,
            "por_watershed": d.por_watershed,
        }
        fila.update(porcentajes_color(hsv_px, reglas))
        for i, v in enumerate(histograma_h(hsv_px)):
            fila[f"hist_h_{i:02d}"] = v
        filas.append(fila)

    return pd.DataFrame(filas)


def columnas_ml(reglas: ReglasMadurez) -> list[str]:
    """Columnas usadas como vector de entrada del clasificador ML."""
    return (["H_media", "H_std", "S_media", "V_media", "a_media", "b_media"]
            + reglas.columnas_pct + [f"hist_h_{i:02d}" for i in range(18)])
