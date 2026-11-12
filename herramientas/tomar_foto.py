r"""Toma una foto con el celular (DroidCam) y la guarda en data/imagenes/.

ESPACIO toma la foto; se ve congelada y se elige: g / ENTER = guardar,
r = repetir. Al guardar, el análisis (reconocer la fruta, contar, madurez, tamaño)
se ejecuta SOLO en AgroVision.ipynb y los resultados quedan escritos en el
notebook: no hay que darle "Ejecutar todo" (que volvería a abrir la cámara).
VS Code recarga el notebook solo; si tenía cambios sin guardar, ciérrelo y ábralo.
Con --sin-notebook solo se guarda la foto.

Uso desde la raíz del proyecto:

    .\.venv\Scripts\python herramientas/tomar_foto.py
    .\.venv\Scripts\python herramientas/tomar_foto.py http://IP_DEL_CELULAR:4747/video
    .\.venv\Scripts\python herramientas/tomar_foto.py 1        # DroidCam Client (cámara 1)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agrovision import config as cfg                        # noqa: E402
from agrovision.captura import tomar_foto                   # noqa: E402


def analizar_en_notebook(nombre_foto: str, notebook: Path | None = None) -> Path:
    """Ejecuta AgroVision.ipynb con la foto dada y deja las salidas escritas en el archivo.

    El notebook se ejecuta con TOMAR_FOTO = False y NOMBRE_IMAGEN = la foto, pero solo
    en memoria: el código guardado en el notebook no cambia, solo sus resultados.
    """
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook = notebook or cfg.RAIZ / "AgroVision.ipynb"
    nb = nbformat.read(notebook, as_version=4)
    original = {i: c.source for i, c in enumerate(nb.cells) if c.cell_type == "code"}
    for c in nb.cells:
        if c.cell_type == "code" and re.search(r"^TOMAR_FOTO\s*=", c.source, re.M):
            c.source = re.sub(r"^TOMAR_FOTO\s*=\s*\w+", "TOMAR_FOTO = False", c.source, flags=re.M)
            c.source = re.sub(r"^NOMBRE_IMAGEN\s*=\s*[^#\n]+", f"NOMBRE_IMAGEN = {nombre_foto!r} ",
                              c.source, flags=re.M)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(nb, km=km, timeout=600, resources={"metadata": {"path": str(cfg.RAIZ)}}).execute()
    for i, src in original.items():                    # el código queda como estaba
        nb.cells[i].source = src
    nbformat.write(nb, notebook)
    return notebook


def main() -> int:
    ap = argparse.ArgumentParser(description="Foto con DroidCam (guardar o repetir) y análisis en el notebook")
    ap.add_argument("--sin-notebook", action="store_true", help="solo guardar la foto, sin analizarla")
    ap.add_argument("origen", nargs="?", default=cfg.ORIGEN_CELULAR, help="índice de cámara o URL")
    a = ap.parse_args()
    origen = int(a.origen) if a.origen.isdigit() else a.origen

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg.crear_estructura()
    try:
        ruta = tomar_foto(origen, cfg.DIR_IMAGENES)
    except (RuntimeError, TimeoutError) as e:
        print(f"No llega imagen del celular: {e}")
        print("Con la app DroidCam abierta, use una de estas dos opciones:")
        print("  1) Si la IP de la app cambió, pásela: herramientas/tomar_foto.py http://IP_DEL_CELULAR:4747/video")
        print("  2) Con DroidCam Client en el PC (Start): herramientas/tomar_foto.py 1")
        return 1
    if ruta is None:
        print("Cancelado: no se guardó ninguna foto.")
        return 0
    print(f"Foto guardada: {ruta.name}")
    if a.sin_notebook:
        print("Ahora ejecute el notebook (TOMAR_FOTO = False para no abrir la cámara otra vez).")
        return 0
    print("Analizando la foto...")
    try:                                                # resumen en la terminal (mismas funciones del notebook)
        from agrovision import resumen
        analisis = resumen.analizar_foto(ruta)
        print(resumen.texto_resumen(analisis))
        resumen.guardar_resumen(analisis)
    except Exception as e:                              # noqa: BLE001
        print(f"No se pudo calcular el resumen ({type(e).__name__}: {e}).")
    print("Escribiendo el análisis completo en AgroVision.ipynb (unos segundos)...")
    try:
        nb = analizar_en_notebook(ruta.name)
    except Exception as e:                              # noqa: BLE001 - se avisa y se sigue
        print(f"No se pudo ejecutar el notebook solo ({type(e).__name__}: {e}).")
        print("Ejecútelo a mano con TOMAR_FOTO = False.")
        return 1
    print(f"Listo: el análisis completo (gráficas) está en {nb.name} (si VS Code no lo recarga, ciérrelo y ábralo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
