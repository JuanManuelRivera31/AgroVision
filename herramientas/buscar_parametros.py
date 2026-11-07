r"""Búsqueda aleatoria de parámetros de segmentación para un perfil de fruta.

Puntúa cada combinación con DOS medidas y se queda con las que van bien en
ambas, para no sobreajustar a una sola fotografía:

* ``F1_real``: contra ``data/anotaciones_cajas.csv`` (la foto real).
* ``F1_sint``: contra la escena sintética de esa fruta en ``tests/sinteticos.py``,
  donde la respuesta correcta se conoce de antemano.

El espacio de búsqueda es RELATIVO al perfil (factores sobre sus valores y
desplazamientos del tono), así que sirve igual para un tomate que para un banano.

Uso desde la raíz del proyecto:

    .\.venv\Scripts\python herramientas/buscar_parametros.py 400
    .\.venv\Scripts\python herramientas/buscar_parametros.py 400 --fruta banano --imagen bananos.jpg

Escribe el ranking en ``resultados/estadisticas/busqueda_parametros_<fruta>.csv``
y la mejor combinación en ``resultados/estadisticas/mejor_<fruta>.json``. Va
imprimiendo el avance (se puede interrumpir con Ctrl+C sin perder nada).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))

from agrovision import config as cfg          # noqa: E402
from agrovision import evaluacion as ev       # noqa: E402
from agrovision import segmentacion as seg    # noqa: E402
from banco import evaluar                     # noqa: E402

# Factores que multiplican el valor del perfil.
FACTORES = {
    "circularidad_min": [0.9, 1.0, 1.1, 1.2],
    "circularidad_min_verde": [0.75, 0.85, 1.0],
    "solidez_min": [0.97, 1.0, 1.03, 1.07],
    "solidez_min_verde": [0.9, 0.95, 1.0],
    "ajuste_elipse_min_verde": [0.83, 0.9, 1.0],
    "area_min_frac": [1.0, 2.5, 5.0, 7.5],
    "kernel_cierre": [1.0, 1.4, 1.9],
    "kernel_apertura_verde": [1.0, 1.4, 1.7],
    "textura_max_verde": [0.67, 1.0, 1.45],
    "ventana_picos_frac": [0.5, 0.67, 0.83, 1.0],
    "umbral_dist_watershed": [0.64, 0.82, 1.0],
    "solidez_para_watershed": [1.0, 1.05, 1.1],
}
KERNELS = {"kernel_cierre", "kernel_apertura_verde"}
FRACCIONES = {"circularidad_min", "circularidad_min_verde", "solidez_min", "solidez_min_verde",
              "ajuste_elipse_min_verde", "umbral_dist_watershed", "solidez_para_watershed"}
# Desplazamientos (en unidades de H) de los bordes de tono de cada clase de color.
DESPLAZAMIENTOS_H = [-4, -2, 0, 0, 2, 4]
UMBRAL_CANDIDATO = 0.40   # solo se evalúa en sintético si la foto real ya va bien


def muestrear(base: cfg.ParametrosSegmentacion, rng: random.Random) -> dict:
    cambios = {}
    for k, factores in FACTORES.items():
        v = getattr(base, k) * rng.choice(factores)
        if k in KERNELS:
            v = max(3, int(round(v)) | 1)
        elif k in FRACCIONES:
            v = round(min(v, 0.99), 3)
        else:
            v = round(v, 4)
        cambios[k] = v
    colores = {}
    for nombre, rangos in base.colores.items():
        nuevos = []
        for bajo, alto in rangos:
            # Un borde en 0 o 179 es la vuelta del tono (rojo): no se mueve.
            hb = bajo[0] if bajo[0] == 0 else int(np.clip(bajo[0] + rng.choice(DESPLAZAMIENTOS_H), 0, 179))
            ha = alto[0] if alto[0] == 179 else int(np.clip(alto[0] + rng.choice(DESPLAZAMIENTOS_H), 0, 179))
            nuevos.append([[hb, *bajo[1:]], [max(ha, hb), *alto[1:]]])
        colores[nombre] = nuevos
    cambios["colores"] = colores
    return cambios


def main(n_iter: int, fruta: str, imagen: str, semilla: int = 7):
    import sinteticos as sin
    img_sin, gt_sin = sin.escena_con_cajas(fruta)
    base = cfg.cargar_perfil(fruta).segmentacion

    def f1_sintetico(cambios):
        res = seg.segmentar(img_sin, cfg.cargar_perfil(fruta, **cambios).segmentacion)
        pr = ev.precision_recall([d.bbox for d in res.detecciones], gt_sin, 0.5)
        return 0.0 if np.isnan(pr["F1"]) else pr["F1"]

    rng = random.Random(semilla)
    filas = []
    try:
        for it in range(n_iter):
            cambios = muestrear(base, rng)
            try:
                pr, _, _ = evaluar(cambios, fruta, imagen, verbose=False)
            except Exception:
                continue
            f1r = 0.0 if np.isnan(pr["F1"]) else pr["F1"]
            if f1r >= UMBRAL_CANDIDATO:
                filas.append({"F1_real": f1r, "F1_sint": f1_sintetico(cambios),
                              "TP": pr["TP"], "FP": pr["FP"], "FN": pr["FN"],
                              "parametros": json.dumps(cambios)})
            if (it + 1) % 25 == 0:
                print(f"  {it + 1}/{n_iter} evaluadas, {len(filas)} candidatas", flush=True)
    except KeyboardInterrupt:
        print("\ninterrumpido: se guarda lo encontrado hasta ahora")

    if not filas:
        print("Ninguna combinación superó el umbral.")
        return
    df = pd.DataFrame(filas)
    df["puntaje"] = df["F1_real"] + df["F1_sint"]
    df = df.sort_values(["puntaje", "FP"], ascending=[False, True])
    ruta = cfg.DIR_RES_ESTADISTICAS / f"busqueda_parametros_{fruta}.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\n{len(df)} candidatas guardadas en {ruta.relative_to(cfg.RAIZ)}")
    print(df.head(8)[["F1_real", "F1_sint", "TP", "FP", "FN"]].to_string(index=False))
    mejor = cfg.DIR_RES_ESTADISTICAS / f"mejor_{fruta}.json"
    mejor.write_text(json.dumps(json.loads(df.iloc[0]["parametros"]), indent=2), encoding="utf-8")
    print(f"\nMejor combinación guardada en {mejor.relative_to(cfg.RAIZ)}. Para verla:")
    print(f"  .\\.venv\\Scripts\\python herramientas/banco.py {mejor.relative_to(cfg.RAIZ)} "
          f"--fruta {fruta} --imagen {imagen} --vista vista.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Búsqueda aleatoria de parámetros de segmentación")
    ap.add_argument("n", nargs="?", type=int, default=400, help="combinaciones a probar")
    ap.add_argument("--fruta", default="tomate", choices=cfg.listar_perfiles())
    ap.add_argument("--imagen", default="tomates.jpg", help="foto anotada de data/imagenes/")
    a = ap.parse_args()
    main(a.n, a.fruta, a.imagen)
