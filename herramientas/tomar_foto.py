r"""Toma una foto con el celular (DroidCam) y la analiza sola.

Abre la cámara en vivo; con ESPACIO toma la foto y muestra los frutos detectados,
el conteo y la madurez. Guarda la foto en data/imagenes/ y los resultados en
resultados/detecciones/ y resultados/estadisticas/.

Uso desde la raíz del proyecto:

    .\.venv\Scripts\python herramientas/tomar_foto.py
    .\.venv\Scripts\python herramientas/tomar_foto.py http://IP_DEL_CELULAR:4747/video
    .\.venv\Scripts\python herramientas/tomar_foto.py 1 --fruta banano  # DroidCam Client (cámara 1)

Teclas: ESPACIO = tomar foto, q = salir.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agrovision import caracteristicas as car               # noqa: E402
from agrovision import clasificacion as clas                # noqa: E402
from agrovision import config as cfg                        # noqa: E402
from agrovision import estereo as est                       # noqa: E402
from agrovision import segmentacion as seg                  # noqa: E402
from agrovision import visualizacion as vis                 # noqa: E402
from agrovision.captura import FuenteCamara, capturar_foto, guardar_foto  # noqa: E402
from agrovision.procesamiento import guardar_imagen         # noqa: E402

TECLA_ESC = 27


def analizar(foto: np.ndarray, perfil, altura_cm: float | None = None):
    """Detecta, cuenta, clasifica y mide. Devuelve (tabla, imagen dibujada, conteo por etapa)."""
    p, reglas = perfil.segmentacion, perfil.madurez
    lado = max(foto.shape[:2])
    if lado < p.lado_max:
        # Los parámetros de forma están en píxeles a lado_max y DroidCam suele dar 640x480.
        f = p.lado_max / lado
        foto = cv2.resize(foto, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    res = seg.segmentar(foto, p)
    df = clas.aplicar_reglas(car.extraer_caracteristicas(res, reglas), reglas)
    vista = vis.dibujar_detecciones(res.imagen, res.detecciones, df, reglas)
    if altura_cm and len(df):
        w, h = foto.shape[1], foto.shape[0]
        fov = est.fov_horizontal_desde_f35(cfg.F35_CELULAR)
        f_px = est.calibracion_por_fov((max(w, h), min(w, h)), fov).K[0, 0]
        df = df.merge(est.medir_por_altura(df, altura_cm * 10, f_px), on="id")
        vista = vis.dibujar_diametros(vista, res.detecciones, df)
    conteos = {c: int((df["categoria"] == c).sum()) if len(df) else 0 for c in reglas.categorias}
    return df, vista, conteos


def ajustar(img: np.ndarray, ancho: int = 1000) -> np.ndarray:
    return cv2.resize(img, None, fx=ancho / img.shape[1], fy=ancho / img.shape[1])


def con_franja(img: np.ndarray, texto: str) -> np.ndarray:
    # cv2.putText no dibuja tildes.
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    franja = np.zeros((46, img.shape[1], 3), np.uint8)
    cv2.putText(franja, texto, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([franja, img])


def procesar_foto(foto: np.ndarray, perfil, altura_cm: float | None) -> tuple[str, np.ndarray]:
    nombre = guardar_foto(foto, cfg.DIR_IMAGENES).stem
    df, vista, conteos = analizar(foto, perfil, altura_cm)
    guardar_imagen(cfg.DIR_RES_DETECCIONES / f"{nombre}_detecciones.png", vista)
    df.to_csv(cfg.DIR_RES_ESTADISTICAS / f"{nombre}_caracteristicas.csv", index=False, encoding="utf-8-sig")
    resumen = " | ".join([f"TOTAL {len(df)}", *(f"{c} {n}" for c, n in conteos.items())])
    print(f"{nombre}: {resumen}")
    return nombre, con_franja(ajustar(vista), f"{resumen}   (tecla: otra foto, q: salir)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Foto con DroidCam + análisis automático")
    ap.add_argument("origen", nargs="?", default=cfg.ORIGEN_CELULAR, help="índice de cámara o URL")
    ap.add_argument("--fruta", default="tomate", help="perfil de perfiles/<fruta>.json")
    ap.add_argument("--altura", type=float, default=cfg.ALTURA_CELULAR_CM,
                    help="altura del lente sobre la mesa en cm")
    a = ap.parse_args()
    origen = int(a.origen) if a.origen.isdigit() else a.origen

    cfg.crear_estructura()
    perfil = cfg.cargar_perfil(a.fruta)
    try:
        camara = FuenteCamara(origen, "celular", resolucion=(1280, 720))
        camara.ultimo(espera_max=8.0)
    except (RuntimeError, TimeoutError) as e:
        print(f"No llega imagen del celular: {e}\n"
              "Con la app DroidCam abierta, use una de estas dos opciones:\n"
              "  1) Si la IP de la app cambió, pásela: herramientas/tomar_foto.py http://IP_DEL_CELULAR:4747/video\n"
              "  2) Con DroidCam Client en el PC (Start): herramientas/tomar_foto.py 1")
        return 1

    ventana = "AgroVision - ESPACIO: tomar foto, q: salir"
    ultimo = None
    try:
        while True:
            frame = capturar_foto(camara, ventana)
            if frame is None:
                break
            ultimo, vista = procesar_foto(frame, perfil, a.altura)
            cv2.imshow(ventana, vista)
            if (cv2.waitKey(0) & 0xFF) in (ord("q"), TECLA_ESC):
                break
    finally:
        camara.cerrar()
        cv2.destroyAllWindows()
    if ultimo:
        print(f"Para verla en el notebook: NOMBRE_IMAGEN = \"{ultimo}.png\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
