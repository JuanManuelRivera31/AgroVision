r"""Revisión de las fotos del tablero ANTES de calibrar.

Para cada foto dice si se encontró el patrón y si tiene el mismo tamaño y
orientación que las demás. Luego calibra con las válidas y revisa lo que decide
si la calibración sirve:

* RMS de reproyección (< 0.5 px ideal, < 1 px aceptable).
* Fotos con error muy por encima del resto (tablero movido, borroso o doblado).
* Cobertura: en qué zonas del encuadre (rejilla 3x3) apareció el tablero. La
  distorsión del lente es mayor en las esquinas: si el tablero nunca pasó por
  ahí, la calibración no la corrige.
* Variedad de inclinación: con fotos todas de frente, la distancia focal queda
  mal determinada.

Al final da un veredicto: APTA, USABLE o NO APTA.

Uso desde la raíz del proyecto:

    .\.venv\Scripts\python herramientas/revisar_tablero.py
    .\.venv\Scripts\python herramientas/revisar_tablero.py data/calibracion/celular --cuadro 24.5
    .\.venv\Scripts\python herramientas/revisar_tablero.py --vista resultados/cobertura.png

Genera también el tablero para imprimir (PDF a tamaño real) con --pdf.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agrovision import config as cfg            # noqa: E402
from agrovision import estereo as est           # noqa: E402
from agrovision.procesamiento import cargar_imagen   # noqa: E402

MIN_VISTAS = 10
RMS_APTA, RMS_USABLE = 0.5, 1.0
COBERTURA_MIN = 7          # de las 9 zonas del encuadre
INCLINACION_MIN = 15.0     # grados: al menos una foto debe estar así de inclinada


def revisar(rutas: list[Path], patron: cfg.PatronCalibracion, verbose: bool = False) -> dict:
    """Revisa las fotos del tablero. Devuelve un dict con la tabla por foto
    (``fotos``), la ``calibracion`` (o None), ``cobertura``, ``inclinacion_max``,
    ``veredicto`` y ``consejos``."""
    tamanos, esquinas = {}, {}
    for ruta in rutas:
        img = cargar_imagen(ruta)
        tamanos[ruta.name] = (img.shape[1], img.shape[0])
        ok, esq = est.detectar_tablero(img, patron)
        esquinas[ruta.name] = esq if ok else None
        if verbose:
            print(f"  {'[ok]        ' if ok else '[sin patrón]'} {ruta.name}", flush=True)

    # El tamaño de referencia es el más común: una foto vertical suelta no manda.
    tam = Counter(tamanos.values()).most_common(1)[0][0] if tamanos else None
    filas = []
    for nombre in tamanos:
        if tamanos[nombre] != tam:
            estado = "otro tamaño u orientación"
        elif esquinas[nombre] is None:
            estado = "sin patrón"
        else:
            estado = "ok"
        filas.append({"archivo": nombre, "estado": estado, "error_px": np.nan,
                      "inclinacion_grados": np.nan})
    fotos = pd.DataFrame(filas, columns=["archivo", "estado", "error_px", "inclinacion_grados"])
    validas = fotos.index[fotos["estado"] == "ok"].tolist()

    consejos = []
    n_otro = int((fotos["estado"] == "otro tamaño u orientación").sum())
    if n_otro:
        consejos.append(f"{n_otro} foto(s) con otro tamaño u orientación: tome todas igual "
                        f"({tam[0]}x{tam[1]}), sin girar el celular ni cambiar de lente.")
    if (fotos["estado"] == "sin patrón").any():
        consejos.append("Fotos sin patrón: el tablero debe verse COMPLETO, enfocado y sin reflejos.")

    inf = {"fotos": fotos, "tam": tam, "calibracion": None, "cobertura": 0,
           "inclinacion_max": 0.0, "veredicto": "NO APTA", "consejos": consejos}
    if len(validas) < MIN_VISTAS:
        consejos.append(f"Solo {len(validas)} fotos válidas: se necesitan al menos {MIN_VISTAS} "
                        "(mejor 15-25).")
        return inf

    obj = est.puntos_objeto(patron)
    pts = [esquinas[fotos.at[i, "archivo"]] for i in validas]
    cal = est.calibrar_desde_puntos([obj] * len(pts), pts, tam,
                                    [fotos.at[i, "archivo"] for i in validas])
    inf["calibracion"] = cal
    for i, e, p in zip(validas, cal.errores_vista, pts):
        fotos.at[i, "error_px"] = round(e, 3)
        _, rvec, _ = cv2.solvePnP(obj, p, cal.K, cal.dist)
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        fotos.at[i, "inclinacion_grados"] = round(float(np.degrees(np.arccos(abs(normal[2])))), 1)

    # Fotos que se salen del resto (más del doble de la mediana y > 1 px).
    errores = fotos.loc[validas, "error_px"]
    malas = fotos.loc[validas][(errores > 2 * errores.median()) & (errores > 1.0)]["archivo"]
    if len(malas):
        consejos.append(f"Error alto en {', '.join(malas)}: quítelas y vuelva a revisar "
                        "(suele ser foto movida o tablero doblado).")

    # Cobertura del encuadre en una rejilla 3x3. Una zona cuenta si entre todas
    # las fotos cayeron en ella al menos la mitad de las esquinas de un tablero
    # (que el borde del tablero la roce no basta).
    todas = np.concatenate([p.reshape(-1, 2) for p in pts])
    zonas = Counter((min(int(3 * x / tam[0]), 2), min(int(3 * y / tam[1]), 2)) for x, y in todas)
    inf["cobertura"] = sum(n >= len(obj) / 2 for n in zonas.values())
    if inf["cobertura"] < COBERTURA_MIN:
        consejos.append(f"El tablero solo pasó por {inf['cobertura']} de 9 zonas del encuadre: "
                        "tome fotos con el tablero cerca de las esquinas y los bordes.")

    inf["inclinacion_max"] = float(fotos.loc[validas, "inclinacion_grados"].max())
    if inf["inclinacion_max"] < INCLINACION_MIN:
        consejos.append(f"Todas las fotos son casi de frente (máx. {inf['inclinacion_max']:.0f}°): "
                        "incline el celular hasta unos 30° en algunas.")

    if cal.rms > RMS_USABLE:
        inf["veredicto"] = "NO APTA"
        consejos.append(f"RMS = {cal.rms:.2f} px (> {RMS_USABLE}): repita las fotos con el "
                        "enfoque bloqueado y el tablero bien plano.")
    elif cal.rms > RMS_APTA or inf["cobertura"] < COBERTURA_MIN \
            or inf["inclinacion_max"] < INCLINACION_MIN:
        inf["veredicto"] = "USABLE"
    else:
        inf["veredicto"] = "APTA"
    return inf


def dibujar_cobertura(inf: dict, rutas: list[Path], patron, ruta_salida: Path) -> None:
    """Imagen con el contorno del tablero de cada foto válida sobre la rejilla 3x3."""
    w, h = inf["tam"]
    lienzo = np.full((h, w, 3), 255, np.uint8)
    for k in (1, 2):
        cv2.line(lienzo, (k * w // 3, 0), (k * w // 3, h), (200, 200, 200), 2)
        cv2.line(lienzo, (0, k * h // 3), (w, k * h // 3), (200, 200, 200), 2)
    ok = set(inf["fotos"].loc[inf["fotos"]["estado"] == "ok", "archivo"])
    c, f = patron.esquinas
    for ruta in rutas:
        if ruta.name not in ok:
            continue
        _, esq = est.detectar_tablero(cargar_imagen(ruta), patron)
        p = esq.reshape(f, c, 2)
        marco = np.array([p[0, 0], p[0, -1], p[-1, -1], p[-1, 0]], np.int32)
        cv2.polylines(lienzo, [marco], True, (0, 140, 0), max(2, w // 400))
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), lienzo)


def main():
    ap = argparse.ArgumentParser(description="Revisa las fotos del tablero antes de calibrar")
    ap.add_argument("carpeta", nargs="?", default=str(cfg.DIR_CALIBRACION / "celular"))
    ap.add_argument("--cuadro", type=float, default=25.0, help="lado del cuadro medido (mm)")
    ap.add_argument("--vista", help="guardar la imagen de cobertura en esta ruta")
    ap.add_argument("--pdf", help="solo generar el tablero imprimible en esta ruta y salir")
    a = ap.parse_args()
    patron = cfg.PatronCalibracion((9, 6), a.cuadro)

    if a.pdf:
        est.figura_tablero_imprimible(patron).savefig(a.pdf)
        print(f"Tablero guardado en {a.pdf}")
        return

    rutas = cfg.listar_imagenes(Path(a.carpeta))
    if not rutas:
        print(f"No hay fotos en {a.carpeta}")
        return
    print(f"Revisando {len(rutas)} fotos de {a.carpeta} (puede tardar con fotos grandes)...")
    inf = revisar(rutas, patron, verbose=True)
    with pd.option_context("display.width", 120):
        print("\n" + inf["fotos"].to_string(index=False))
    if inf["calibracion"] is not None:
        print("\n" + inf["calibracion"].resumen())
        print(f"Cobertura: {inf['cobertura']}/9 zonas | inclinación máxima: "
              f"{inf['inclinacion_max']:.0f}°")
    if a.vista and inf["calibracion"] is not None:
        dibujar_cobertura(inf, rutas, patron, Path(a.vista))
        print(f"Cobertura dibujada en {a.vista}")
    print(f"\nVEREDICTO: {inf['veredicto']}")
    for c in inf["consejos"]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
