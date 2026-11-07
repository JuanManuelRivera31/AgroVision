"""Evaluación contra referencias manuales.

Ninguna función produce métricas sin datos de referencia reales. Formatos:

* ``data/anotaciones_cajas.csv``: imagen, x, y, w, h, categoria
  (cajas dibujadas a mano, en píxeles de la imagen ORIGINAL).
* ``data/medidas_reales.csv``: imagen, id, diametro_real_mm, profundidad_real_mm
  (medidas con calibrador/regla; ``id`` es el número que asignó el sistema).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def error_conteo(conteo_sistema: int, conteo_manual: int) -> dict:
    err = conteo_sistema - conteo_manual
    return {"conteo_sistema": conteo_sistema, "conteo_manual": conteo_manual,
            "error_absoluto": abs(err), "error_con_signo": err,
            "error_relativo_%": 100 * abs(err) / conteo_manual if conteo_manual else np.nan}


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def emparejar_detecciones(pred: list[tuple], gt: list[tuple], umbral_iou: float = 0.5):
    """Emparejamiento voraz por IoU descendente. Devuelve lista de (i_pred, i_gt, iou)."""
    pares = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
                   reverse=True)
    usados_p, usados_g, res = set(), set(), []
    for v, i, j in pares:
        if v < umbral_iou:
            break
        if i not in usados_p and j not in usados_g:
            usados_p.add(i)
            usados_g.add(j)
            res.append((i, j, v))
    return res


def precision_recall(pred: list[tuple], gt: list[tuple], umbral_iou: float = 0.5) -> dict:
    m = emparejar_detecciones(pred, gt, umbral_iou)
    tp = len(m)
    fp, fn = len(pred) - tp, len(gt) - tp
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * p * r / (p + r) if p and r and not np.isnan(p + r) else np.nan
    return {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "F1": f1,
            "IoU_medio": float(np.mean([v for *_, v in m])) if m else np.nan,
            "emparejamientos": m}


def evaluar_clasificacion(y_real, y_pred, categorias: list[str]) -> dict:
    """``categorias`` fija el orden de filas/columnas (las etapas del perfil)."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    etiquetas = [c for c in categorias if c in set(y_real) | set(y_pred)]
    return {
        "exactitud": accuracy_score(y_real, y_pred),
        "matriz": pd.DataFrame(confusion_matrix(y_real, y_pred, labels=etiquetas),
                               index=[f"real {c}" for c in etiquetas],
                               columns=[f"pred {c}" for c in etiquetas]),
        "reporte": classification_report(y_real, y_pred, labels=etiquetas, zero_division=0),
    }


def error_medidas(estimado: pd.Series, real: pd.Series) -> dict:
    """MAE, RMSE y error relativo medio entre medidas estimadas y reales."""
    ok = estimado.notna() & real.notna()
    e, r = estimado[ok].to_numpy(float), real[ok].to_numpy(float)
    if len(e) == 0:
        return {"n": 0}
    d = e - r
    return {"n": len(e), "MAE": float(np.mean(np.abs(d))),
            "RMSE": float(np.sqrt(np.mean(d ** 2))),
            "error_relativo_medio_%": float(np.mean(np.abs(d) / r) * 100),
            "sesgo": float(np.mean(d))}


class Cronometro:
    """Mide el tiempo de cada etapa del pipeline."""

    def __init__(self):
        self.tiempos: dict[str, float] = {}

    @contextmanager
    def etapa(self, nombre: str):
        t0 = time.perf_counter()
        yield
        self.tiempos[nombre] = self.tiempos.get(nombre, 0) + (time.perf_counter() - t0) * 1000

    def tabla(self) -> pd.DataFrame:
        df = pd.DataFrame({"etapa": list(self.tiempos), "tiempo_ms": list(self.tiempos.values())})
        return pd.concat([df, pd.DataFrame([{"etapa": "TOTAL", "tiempo_ms": df["tiempo_ms"].sum()}])],
                         ignore_index=True)
