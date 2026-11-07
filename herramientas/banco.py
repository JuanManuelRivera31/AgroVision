r"""Banco de medición de la detección contra anotaciones reales.

No forma parte del sistema: es la herramienta que permite ajustar un perfil
con un número en vez de a ojo. Compara las detecciones con las cajas de
``data/anotaciones_cajas.csv`` y reporta precisión / recall / F1 con IoU 0.5.

El argumento opcional es un .json con los parámetros de segmentación a cambiar
sobre el perfil (en ``colores`` basta con las clases que cambian).

Uso desde la raíz del proyecto:

    .\.venv\Scripts\python herramientas/banco.py
    .\.venv\Scripts\python herramientas/banco.py herramientas/perfil_ajustado.json --vista vista.png
    .\.venv\Scripts\python herramientas/banco.py --fruta banano --imagen bananos.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agrovision import config as cfg          # noqa: E402
from agrovision import evaluacion as ev       # noqa: E402
from agrovision import procesamiento as proc  # noqa: E402
from agrovision import segmentacion as seg    # noqa: E402


def cajas_reales(nombre_imagen: str, escala: float) -> tuple[list, list]:
    """Anotaciones de esa imagen, llevadas de píxeles originales a los de trabajo."""
    a = pd.read_csv(cfg.ARCHIVO_ANOTACIONES, encoding="utf-8-sig")
    a = a[a["imagen"] == nombre_imagen]
    if a.empty:
        raise ValueError(f"No hay cajas anotadas para '{nombre_imagen}' en {cfg.ARCHIVO_ANOTACIONES.name}")
    cajas = [tuple(int(round(v * escala)) for v in f)
             for f in a[["x", "y", "w", "h"]].to_numpy()]
    return cajas, list(a["categoria"])


def evaluar(cambios: dict | None = None, fruta: str = "tomate", imagen: str = "tomates.jpg",
            dibujar: str | None = None, verbose: bool = True):
    """Segmenta con el perfil (más los cambios dados) y mide contra las anotaciones."""
    p = cfg.cargar_perfil(fruta, **(cambios or {})).segmentacion
    ruta = cfg.DIR_IMAGENES / imagen
    img, escala = proc.redimensionar(proc.cargar_imagen(ruta), p.lado_max)
    res = seg.segmentar(img, p)
    gt, cats = cajas_reales(imagen, escala)
    pred = [d.bbox for d in res.detecciones]
    pr = ev.precision_recall(pred, gt, 0.5)
    emparejados = {g for _, g, _ in pr["emparejamientos"]}

    if verbose:
        print(f"  detecciones={len(pred)}  TP={pr['TP']} FP={pr['FP']} FN={pr['FN']}  "
              f"P={pr['precision']:.2f} R={pr['recall']:.2f} F1={pr['F1']:.2f}")
        faltan = [f"#{i + 1} {cats[i]}" for i in range(len(gt)) if i not in emparejados]
        if faltan:
            print("   sin detectar:", faltan)

    if dibujar:
        vista = img.copy()
        for i, (x, y, w, h) in enumerate(gt):          # verde = encontrado, rojo = perdido
            color = (0, 255, 0) if i in emparejados else (0, 0, 255)
            cv2.rectangle(vista, (x, y), (x + w, y + h), color, 3)
            cv2.putText(vista, f"GT{i + 1} {cats[i]}", (x + 4, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for d in res.detecciones:                       # amarillo = lo que detectó
            x, y, w, h = d.bbox
            cv2.rectangle(vista, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(vista, f"d{d.id}", (x + 4, y + h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        proc.guardar_imagen(Path(dibujar), vista)
        print("   vista guardada en", dibujar)
    return pr, res, p


def leer_cambios(ruta: str) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cambios", nargs="?", help=".json con parámetros a cambiar sobre el perfil")
    ap.add_argument("--fruta", default="tomate", choices=cfg.listar_perfiles())
    ap.add_argument("--imagen", default="tomates.jpg", help="foto de data/imagenes/")
    ap.add_argument("--vista", help="dónde guardar la vista comparativa (.png)")
    a = ap.parse_args()
    cambios = leer_cambios(a.cambios) if a.cambios else {}
    print(f"fruta: {a.fruta} | imagen: {a.imagen} | cambios:", cambios or "(ninguno)")
    evaluar(cambios, a.fruta, a.imagen, dibujar=a.vista)
