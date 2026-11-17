r"""Analiza una foto YA GUARDADA (sin cámara): reconoce la fruta, cuenta, madurez y tamaño.

    .\.venv\Scripts\python herramientas/analizar_foto.py data/imagenes/mangos.jpg
    .\.venv\Scripts\python herramientas/analizar_foto.py mangos.jpg --sin-ventana

Imprime el resumen en la terminal, guarda la imagen del resultado en
resultados/detecciones/ y (salvo --sin-ventana) la muestra en una ventana.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agrovision import config as cfg, resumen              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Analiza una foto guardada (sin cámara)")
    ap.add_argument("foto", help="ruta de la foto, o solo el nombre si está en data/imagenes")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir la ventana con el resultado")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ruta = Path(a.foto)
    if not ruta.exists():
        ruta = cfg.DIR_IMAGENES / a.foto
    if not ruta.exists():
        print(f"No existe la foto '{a.foto}'. Disponibles en data/imagenes:")
        for p in cfg.listar_imagenes(cfg.DIR_IMAGENES):
            print("  ", p.name)
        return 1

    cfg.crear_estructura()
    analisis = resumen.analizar_foto(ruta)
    print(resumen.texto_resumen(analisis))
    resumen.guardar_resumen(analisis)
    salida = resumen.guardar_imagen_resultado(analisis)
    print(f"\nImagen del resultado: {salida}")
    if not a.sin_ventana:
        import cv2
        img = resumen.imagen_resultado(analisis)
        f = 1100 / img.shape[1]
        ventana = "AgroVision - resultado (cualquier tecla para cerrar)"
        cv2.imshow(ventana, cv2.resize(img, None, fx=f, fy=f))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
