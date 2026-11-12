"""Carga de imágenes y preprocesamiento digital con OpenCV."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def cargar_imagen(ruta: str | Path) -> np.ndarray:
    """Carga una imagen en BGR.

    Se usa ``np.fromfile`` + ``cv2.imdecode`` en lugar de ``cv2.imread`` porque
    ``cv2.imread`` falla en Windows cuando la ruta tiene tildes o eñes.
    ``imdecode`` respeta la orientación EXIF de las fotos del celular.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV no pudo decodificar: {ruta}")
    return img


def guardar_imagen(ruta: str | Path, img: np.ndarray) -> Path:
    """Guarda una imagen (compatible con rutas con tildes en Windows)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ruta.suffix or ".png", img)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen para {ruta}")
    buf.tofile(str(ruta))
    return ruta


def redimensionar(img: np.ndarray, lado_max: int, ampliar: bool = False) -> tuple[np.ndarray, float]:
    """Reescala para que el lado mayor sea ``lado_max``. Devuelve (imagen, escala).

    Las fotos de celular (12 MP o más) se reducen para que el procesamiento sea
    rápido; los parámetros morfológicos están definidos a esta escala. Con
    ``ampliar`` las fotos más pequeñas (imágenes de internet) también se llevan
    a ``lado_max``: si no, un kernel de 15 px es relativamente más grande y los
    frutos vecinos se funden.
    """
    h, w = img.shape[:2]
    escala = lado_max / max(h, w)
    if abs(escala - 1.0) < 1e-6 or (escala > 1.0 and not ampliar):
        return img.copy(), 1.0
    nueva = cv2.resize(img, (round(w * escala), round(h * escala)),
                       interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC)
    return nueva, escala


def a_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """OpenCV trabaja en BGR; Matplotlib espera RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def a_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def a_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def filtro_gaussiano(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Suaviza ruido de alta frecuencia (sensor, textura fina)."""
    return cv2.GaussianBlur(img, (k, k), 0)


def filtro_mediana(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Elimina ruido impulsivo conservando bordes mejor que el gaussiano."""
    return cv2.medianBlur(img, k)


def normalizar_iluminacion(img_bgr: np.ndarray, clip: float = 2.0,
                           rejilla: int = 8) -> np.ndarray:
    """CLAHE sobre el canal L de LAB.

    Ecualiza el contraste localmente sin alterar el color (canales a y b), lo
    que reduce el efecto de sombras en el follaje.
    """
    lab = a_lab(img_bgr)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(rejilla, rejilla))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocesar(img_bgr: np.ndarray, k_mediana: int = 5,
                k_gauss: int = 5) -> dict[str, np.ndarray]:
    """Cadena de preprocesamiento usada por el detector.

    1. CLAHE (iluminación) -> 2. mediana (ruido impulsivo, bordes) ->
    3. gaussiano (suavizado final) -> conversión a HSV y LAB.
    Devuelve todas las etapas para poder visualizarlas.
    """
    ilum = normalizar_iluminacion(img_bgr)
    mediana = filtro_mediana(ilum, k_mediana)
    suave = filtro_gaussiano(mediana, k_gauss)
    return {
        "iluminacion": ilum,
        "mediana": mediana,
        "suavizada": suave,
        "hsv": a_hsv(suave),
        "lab": a_lab(suave),
    }


def desviacion_local(gris: np.ndarray, ventana: int = 9) -> np.ndarray:
    """Desviación estándar local (medida de textura).

    sigma = sqrt(E[x^2] - E[x]^2) calculado con cv2.blur (filtro de caja).
    Los tomates tienen superficie lisa (sigma bajo); las hojas, nervaduras y
    bordes producen sigma alto.
    """
    g = gris.astype(np.float32)
    media = cv2.blur(g, (ventana, ventana))
    media2 = cv2.blur(g * g, (ventana, ventana))
    return np.sqrt(np.maximum(media2 - media * media, 0))
