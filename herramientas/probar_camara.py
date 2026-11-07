r"""Prueba de cámaras antes de calibrar (Modo 3: cámara del PC + celular).

Abre una o dos fuentes, muestra la imagen en vivo y mide lo que decide si la
captura estéreo va a funcionar: resolución, fotogramas por segundo, si la
imagen se congela y, con dos cámaras, el DESFASE entre ellas (el notebook
rechaza las capturas con más de 80 ms).

Fuentes: un número (0 = webcam del PC; DroidCam/Iriun por USB suelen ser 1 o 2)
o una URL (IP Webcam: http://IP_DEL_CELULAR:8080/video, misma red Wi-Fi).

Uso desde la raíz del proyecto:

    .\.venv\Scripts\python herramientas/probar_camara.py --listar
    .\.venv\Scripts\python herramientas/probar_camara.py 1
    .\.venv\Scripts\python herramientas/probar_camara.py http://192.168.1.50:8080/video
    .\.venv\Scripts\python herramientas/probar_camara.py 0 1
    .\.venv\Scripts\python herramientas/probar_camara.py 0 1 --segundos 20 --sin-ventana

Teclas en la ventana: q = salir, g = guardar la imagen actual en resultados/capturas/.

Retraso REAL del celular: apunte el celular a esta misma ventana. El reloj grande
es la hora actual del PC; el que se ve dentro de la imagen del celular es la hora
de hace un momento. La diferencia entre los dos es el retraso de la transmisión.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agrovision import config as cfg            # noqa: E402
from agrovision.captura import FuenteCamara     # noqa: E402
from agrovision.procesamiento import guardar_imagen  # noqa: E402

DESFASE_MAX_MS = 80.0       # el mismo límite que usa capturar_interactivo
SIN_IMAGEN_S = 1.0          # sin fotogramas nuevos en este tiempo = congelada
DIR_CAPTURAS = cfg.RESULTADOS / "capturas"


def interpretar_origen(texto: str) -> int | str:
    return int(texto) if texto.isdigit() else texto


def listar_camaras(max_indice: int = 6) -> list[int]:
    """Prueba los índices locales 0..max_indice-1 y dice cuáles entregan imagen."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    encontradas = []
    for i in range(max_indice):
        c = cv2.VideoCapture(i, backend)
        if c.isOpened():
            ok, f = c.read()
            if ok:
                encontradas.append(i)
                print(f"  índice {i}: OK — {f.shape[1]}x{f.shape[0]}")
            else:
                print(f"  índice {i}: se abre pero no entrega imagen (¿la usa otro programa?)")
        c.release()
    if not encontradas:
        print("  Ninguna cámara local respondió.")
    return encontradas


class Medidor:
    """FPS de una fuente en una ventana deslizante de ~2 s, a partir de su contador."""

    def __init__(self, fuente: FuenteCamara):
        self.fuente = fuente
        self.historia = deque(maxlen=120)
        self.n0, self.t0 = fuente.n_fotogramas, time.perf_counter()

    def fps(self) -> float:
        ahora = time.perf_counter()
        self.historia.append((ahora, self.fuente.n_fotogramas))
        while len(self.historia) > 2 and ahora - self.historia[0][0] > 2.0:
            self.historia.popleft()
        (ta, na), (tb, nb) = self.historia[0], self.historia[-1]
        return (nb - na) / (tb - ta) if tb > ta else 0.0

    def fps_medio(self) -> float:
        dt = time.perf_counter() - self.t0
        return (self.fuente.n_fotogramas - self.n0) / dt if dt > 0 else 0.0


def aviso_fps(origen: int | str, fps: float) -> str:
    """Con pocos FPS cada imagen puede tener hasta 1/fps de antigüedad: eso sube el desfase."""
    if fps >= 15:
        return ""
    if isinstance(origen, int):
        return (f"  <- pocos FPS: cada imagen puede tener hasta {1000 / max(fps, 1):.0f} ms de antigüedad. "
                "Pruebe --resolucion 640x480 (muchas webcams solo dan 30 FPS a esa resolución).")
    return "  <- pocos FPS: revise la señal Wi-Fi o baje la resolución en la app del celular."


def resumen_desfases(desfases_ms: list[float]) -> dict:
    d = np.asarray(desfases_ms, float)
    if len(d) == 0:
        return {"n": 0}
    return {"n": len(d), "mediana_ms": float(np.median(d)), "p90_ms": float(np.percentile(d, 90)),
            "aptas_%": float(np.mean(d <= DESFASE_MAX_MS) * 100)}


def veredicto_par(r: dict) -> str:
    if r["n"] == 0:
        return "sin datos"
    if r["p90_ms"] <= DESFASE_MAX_MS:
        return "APTO para el Modo 3: casi todas las capturas pasan el límite de 80 ms."
    if r["mediana_ms"] <= DESFASE_MAX_MS:
        return ("USABLE: la mayoría de las capturas pasan, pero se rechazarán algunas. "
                "Escena estática y paciencia al pulsar 'c'.")
    return ("NO APTO: el desfase típico supera 80 ms. Pruebe el celular por USB "
            "(DroidCam/Iriun) o baje la resolución de la app.")


def componer_vista(imagenes: list, lineas: list[str], escala: float) -> np.ndarray:
    h = min(i.shape[0] for i in imagenes)
    vista = np.hstack([cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imagenes])
    vista = cv2.resize(vista, None, fx=escala, fy=escala)
    for k, (texto, color) in enumerate(lineas):
        y = 24 + 24 * k
        cv2.putText(vista, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vista, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    reloj = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    yb = vista.shape[0] - 20
    cv2.putText(vista, reloj, (10, yb), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(vista, reloj, (10, yb), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vista


def probar(origenes: list, segundos: float | None, con_ventana: bool,
           resolucion: tuple[int, int] | None, escala: float) -> dict:
    nombres = ["izquierda", "derecha"] if len(origenes) == 2 else [str(origenes[0])]
    fuentes = []
    try:
        for o, n in zip(origenes, nombres):
            print(f"Abriendo {n}: {o} ...", flush=True)
            fuentes.append(FuenteCamara(o, n, resolucion))
        for f in fuentes:
            img, _ = f.ultimo(espera_max=8.0)
            print(f"  {f.nombre}: primera imagen recibida, {img.shape[1]}x{img.shape[0]}")
        medidores = [Medidor(f) for f in fuentes]
        desfases, congeladas = [], {f.nombre: 0 for f in fuentes}
        ventana = "AgroVision - prueba de camaras (q: salir, g: guardar)"
        inicio = time.perf_counter()

        while segundos is None or time.perf_counter() - inicio < segundos:
            capturas = [f.ultimo() for f in fuentes]
            ahora = time.perf_counter()
            lineas = []
            for f, m, (img, t) in zip(fuentes, medidores, capturas):
                edad = ahora - t
                if edad > SIN_IMAGEN_S:
                    congeladas[f.nombre] += 1
                color = (0, 0, 255) if edad > SIN_IMAGEN_S else (0, 220, 0)
                lineas.append((f"{f.nombre}: {img.shape[1]}x{img.shape[0]} | {m.fps():4.1f} fps | "
                               f"ultima imagen hace {edad * 1000:4.0f} ms"
                               + ("  CONGELADA" if edad > SIN_IMAGEN_S else ""), color))
            if len(capturas) == 2:
                d = abs(capturas[0][1] - capturas[1][1]) * 1000
                desfases.append(d)
                ok = d <= DESFASE_MAX_MS
                lineas.append((f"desfase entre camaras: {d:4.0f} ms ({'OK' if ok else '> 80 ms, se rechazaria'})",
                               (0, 220, 0) if ok else (0, 0, 255)))

            if con_ventana:
                cv2.imshow(ventana, componer_vista([img for img, _ in capturas], lineas, escala))
                tecla = cv2.waitKey(15) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("g"):
                    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    for f, (img, _) in zip(fuentes, capturas):
                        ruta = guardar_imagen(DIR_CAPTURAS / f"prueba_{f.nombre}_{marca}.png", img)
                        print("  guardada", ruta.relative_to(cfg.RAIZ))
            else:
                time.sleep(0.03)

        informe = {"duracion_s": time.perf_counter() - inicio, "fuentes": []}
        print("\n=== Resumen ===")
        for f, m in zip(fuentes, medidores):
            img, _ = f.ultimo()
            fps = m.fps_medio()
            informe["fuentes"].append({"nombre": f.nombre, "origen": f.origen, "fps": fps,
                                       "resolucion": (img.shape[1], img.shape[0])})
            print(f"{f.nombre} ({f.origen}): {img.shape[1]}x{img.shape[0]}, {fps:.1f} fps de media"
                  + aviso_fps(f.origen, fps))
            if congeladas[f.nombre]:
                print(f"  AVISO: estuvo sin imagen nueva más de {SIN_IMAGEN_S:.0f} s en algún momento.")
        if len(fuentes) == 2:
            r = resumen_desfases(desfases)
            informe["desfase"] = r
            if r["n"]:
                print(f"Desfase: mediana {r['mediana_ms']:.0f} ms, p90 {r['p90_ms']:.0f} ms, "
                      f"{r['aptas_%']:.0f} % de las lecturas <= {DESFASE_MAX_MS:.0f} ms")
            print("Veredicto:", veredicto_par(r))
        return informe
    finally:
        for f in fuentes:
            f.cerrar()
        if con_ventana:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prueba de cámaras (resolución, FPS, desfase)")
    ap.add_argument("origenes", nargs="*", help="1 o 2 fuentes: índice (0, 1...) o URL")
    ap.add_argument("--listar", action="store_true", help="probar los índices locales 0-5")
    ap.add_argument("--segundos", type=float, help="duración de la prueba (por defecto hasta pulsar q)")
    ap.add_argument("--sin-ventana", action="store_true", help="solo medir, sin mostrar imagen")
    ap.add_argument("--resolucion", default="1280x720", help="pedida a las cámaras locales, p. ej. 1920x1080")
    ap.add_argument("--escala", type=float, default=0.6, help="tamaño de la ventana")
    a = ap.parse_args()

    if a.listar:
        print("Cámaras locales:")
        listar_camaras()
        sys.exit(0)
    if not 1 <= len(a.origenes) <= 2:
        ap.error("indique 1 o 2 fuentes (o --listar)")
    if a.sin_ventana and a.segundos is None:
        a.segundos = 10.0
    ancho, alto = (int(v) for v in a.resolucion.lower().split("x"))
    try:
        probar([interpretar_origen(o) for o in a.origenes], a.segundos, not a.sin_ventana,
               (ancho, alto), a.escala)
    except (RuntimeError, TimeoutError) as e:
        print("\nERROR:", e)
        print("Revise: ¿app del celular abierta y transmitiendo? ¿misma red Wi-Fi? "
              "¿la URL termina en /video? ¿otro programa usa la cámara?")
        sys.exit(1)
