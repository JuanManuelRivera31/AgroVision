"""Modo 3: captura desde dos cámaras físicas (cámara del PC + celular).

Fuentes soportadas por ``FuenteCamara``:
* Entero (0, 1, ...): cámara local (la integrada del PC suele ser 0).
* URL: transmisión del celular compatible con OpenCV, por ejemplo la app
  "IP Webcam" (Android): ``http://IP_DEL_CELULAR:8080/video``. El PC y el
  celular deben estar en la misma red Wi-Fi. También sirve DroidCam/Iriun,
  que crean una cámara virtual con índice entero.

Para reducir el desfase temporal cada fuente se lee en un hilo que siempre
conserva el ÚLTIMO fotograma (los buffers de red/USB acumulan fotogramas
viejos). Aun así no hay sincronización por hardware: el desfase medido se
reporta y las pruebas deben hacerse con escenas estáticas.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .procesamiento import guardar_imagen


class FuenteCamara:
    def __init__(self, origen: int | str, nombre: str = "",
                 resolucion: tuple[int, int] | None = None):
        self.origen = origen
        self.nombre = nombre or str(origen)
        backend = cv2.CAP_DSHOW if isinstance(origen, int) and _es_windows() else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(origen, backend)
        if resolucion and isinstance(origen, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente '{self.nombre}' ({origen}).")
        self._frame = None
        self._t = 0.0
        self.n_fotogramas = 0          # fotogramas recibidos desde que se abrió
        self._lock = threading.Lock()
        self._activo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _leer(self):
        while self._activo:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._t = frame, time.perf_counter()
                    self.n_fotogramas += 1
            else:
                time.sleep(0.01)

    def ultimo(self, espera_max: float = 3.0):
        """Último fotograma disponible y su instante de llegada."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < espera_max:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy(), self._t
            time.sleep(0.01)
        raise TimeoutError(f"La fuente '{self.nombre}' no entregó imágenes.")

    def cerrar(self):
        self._activo = False
        self._hilo.join(timeout=1)
        self.cap.release()


def _es_windows() -> bool:
    import os
    return os.name == "nt"


class ParEstereo:
    """Dos fuentes como cámara izquierda/derecha (intercambiables)."""

    def __init__(self, origen_izq, origen_der, nombres=("izquierda", "derecha"),
                 resolucion=None):
        self.izq = FuenteCamara(origen_izq, nombres[0], resolucion)
        try:
            self.der = FuenteCamara(origen_der, nombres[1], resolucion)
        except Exception:
            self.izq.cerrar()
            raise

    def intercambiar(self):
        self.izq, self.der = self.der, self.izq

    def capturar(self):
        """Devuelve (img_izq, img_der, desfase_ms)."""
        fi, ti = self.izq.ultimo()
        fd, td = self.der.ultimo()
        return fi, fd, abs(ti - td) * 1000

    def cerrar(self):
        self.izq.cerrar()
        self.der.cerrar()


def capturar_interactivo(par: ParEstereo, dir_izq: Path, dir_der: Path,
                         prefijo: str = "par", desfase_max_ms: float = 80.0,
                         escala_vista: float = 0.5) -> int:
    """Ventana de OpenCV para capturar pares.

    Teclas: ``c`` captura, ``s`` intercambia izquierda/derecha, ``q`` sale.
    Se rechazan capturas con desfase mayor a ``desfase_max_ms``.
    Devuelve el número de pares guardados.
    """
    n = 0
    ventana = "AgroVision - captura estereo (c: capturar, s: intercambiar, q: salir)"
    try:
        while True:
            fi, fd, desfase = par.capturar()
            h = min(fi.shape[0], fd.shape[0])
            vi = cv2.resize(fi, (int(fi.shape[1] * h / fi.shape[0]), h))
            vd = cv2.resize(fd, (int(fd.shape[1] * h / fd.shape[0]), h))
            vista = np.hstack([vi, vd])
            vista = cv2.resize(vista, None, fx=escala_vista, fy=escala_vista)
            color = (0, 200, 0) if desfase <= desfase_max_ms else (0, 0, 255)
            cv2.putText(vista, f"IZQ: {par.izq.nombre} | DER: {par.der.nombre} | "
                        f"desfase {desfase:.0f} ms | guardados {n}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow(ventana, vista)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                par.intercambiar()
            if k == ord("c"):
                if desfase > desfase_max_ms:
                    print(f"Captura rechazada: desfase {desfase:.0f} ms")
                    continue
                marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                guardar_imagen(dir_izq / f"{prefijo}_{marca}.png", fi)
                guardar_imagen(dir_der / f"{prefijo}_{marca}.png", fd)
                n += 1
                print(f"Par {n} guardado (desfase {desfase:.0f} ms)")
    finally:
        cv2.destroyWindow(ventana)
    return n
