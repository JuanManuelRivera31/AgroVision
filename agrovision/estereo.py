"""Visión estereoscópica con OpenCV: calibración, rectificación, disparidad y 3D.

Convenciones
------------
* Cámara 1 = IZQUIERDA, cámara 2 = DERECHA. ``R, T`` transforman puntos del
  sistema de la cámara izquierda al de la derecha (convención de
  ``cv2.stereoCalibrate`` y ``cv2.recoverPose``).
* Las unidades métricas son las del patrón de calibración (milímetros).
* Nunca se inventan profundidades: si no hay calibración válida, solo se
  calcula disparidad relativa (en píxeles).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import ParametrosSGBM, PatronCalibracion
from .procesamiento import cargar_imagen


# ---------------------------------------------------------------------------
# Verificación y armonización de pares
# ---------------------------------------------------------------------------
def verificar_par(img_izq: np.ndarray, img_der: np.ndarray) -> dict:
    """Comprueba resolución y orientación de un par estéreo."""
    hi, wi = img_izq.shape[:2]
    hd, wd = img_der.shape[:2]
    ori = lambda w, h: "horizontal" if w >= h else "vertical"
    avisos = []
    if (hi, wi) != (hd, wd):
        avisos.append(f"Resoluciones distintas: {wi}x{hi} vs {wd}x{hd}. "
                      "Se deben armonizar antes de calibrar/rectificar.")
    if ori(wi, hi) != ori(wd, hd):
        avisos.append("Orientaciones distintas (una vertical y otra horizontal). "
                      "Rote la imagen o repita la foto con la misma orientación.")
    if abs(wi / hi - wd / hd) > 0.01:
        avisos.append("Relaciones de aspecto distintas: se recortará al centro.")
    return {
        "izquierda": f"{wi}x{hi} ({ori(wi, hi)})",
        "derecha": f"{wd}x{hd} ({ori(wd, hd)})",
        "valido_sin_cambios": not avisos,
        "avisos": avisos,
    }


def armonizar(img: np.ndarray, tam: tuple[int, int]) -> np.ndarray:
    """Recorta al centro a la relación de aspecto ``tam=(ancho, alto)`` y reescala.

    Necesario cuando las cámaras son distintas (PC y celular): stereoRectify
    exige el mismo tamaño de imagen. La MISMA transformación debe aplicarse a
    las fotos de calibración y a las de la escena, porque cambia la matriz K.
    """
    w_obj, h_obj = tam
    h, w = img.shape[:2]
    asp_obj, asp = w_obj / h_obj, w / h
    if asp > asp_obj:                      # sobra ancho
        nw = round(h * asp_obj)
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    elif asp < asp_obj:                    # sobra alto
        nh = round(w / asp_obj)
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh, :]
    interp = cv2.INTER_AREA if img.shape[1] > w_obj else cv2.INTER_LINEAR
    return cv2.resize(img, (w_obj, h_obj), interpolation=interp)


# ---------------------------------------------------------------------------
# Patrón de calibración
# ---------------------------------------------------------------------------
def generar_tablero(patron: PatronCalibracion, px_cuadro: int = 100,
                    margen: int = 60) -> np.ndarray:
    """Imagen del tablero de ajedrez para imprimir (esquinas internas + 1 cuadros)."""
    cols, filas = patron.esquinas[0] + 1, patron.esquinas[1] + 1
    tablero = np.full((filas * px_cuadro + 2 * margen,
                       cols * px_cuadro + 2 * margen), 255, np.uint8)
    for f in range(filas):
        for c in range(cols):
            if (f + c) % 2 == 0:
                y, x = margen + f * px_cuadro, margen + c * px_cuadro
                tablero[y:y + px_cuadro, x:x + px_cuadro] = 0
    return tablero


def figura_tablero_imprimible(patron: PatronCalibracion, hoja_mm=(279.4, 215.9)):
    """Figura de Matplotlib del tablero a TAMAÑO REAL (por defecto carta horizontal).

    Guardada como PDF e impresa a "tamaño real" / 100 %, cada cuadro mide
    ``patron.tam_cuadro_mm``. Incluye una regla de 100 mm para comprobar que la
    impresora no reescaló. Cabe también en A4 (297 x 210 mm).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ancho, alto = hoja_mm
    cols, filas = patron.esquinas[0] + 1, patron.esquinas[1] + 1
    lado = patron.tam_cuadro_mm
    if cols * lado > ancho - 20 or filas * lado > alto - 35:
        raise ValueError(f"Un tablero de {cols}x{filas} cuadros de {lado} mm no cabe en la hoja")
    x0 = (ancho - cols * lado) / 2
    y_sup = alto - 12                       # borde superior del tablero (mm)

    fig = plt.figure(figsize=(ancho / 25.4, alto / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis("off")
    for f in range(filas):
        for c in range(cols):
            if (f + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * lado, y_sup - (f + 1) * lado), lado, lado,
                                       facecolor="black", edgecolor="none"))
    # Regla de 100 mm con marcas cada 10 mm.
    y_regla = 14
    ax.plot([x0, x0 + 100], [y_regla, y_regla], color="black", lw=1)
    for i in range(11):
        ax.plot([x0 + 10 * i] * 2, [y_regla, y_regla + (4 if i % 5 == 0 else 2.5)],
                color="black", lw=1)
    ax.text(x0, y_regla - 6, "Esta línea debe medir 100 mm", fontsize=8)
    ax.text(x0 + 110, y_regla + 1,
            f"Tablero {patron.esquinas[0]}x{patron.esquinas[1]} esquinas internas, "
            f"cuadro de {lado:g} mm.\nImprimir a \"tamaño real\" / 100 % (no \"ajustar a página\").\n"
            "Medir un cuadro con regla y poner el valor en PATRON.tam_cuadro_mm.",
            fontsize=8, va="center")
    return fig


def puntos_objeto(patron: PatronCalibracion) -> np.ndarray:
    """Coordenadas 3D (mm) de las esquinas internas, con Z = 0 en el plano del patrón."""
    c, f = patron.esquinas
    obj = np.zeros((c * f, 3), np.float32)
    obj[:, :2] = np.mgrid[0:c, 0:f].T.reshape(-1, 2) * patron.tam_cuadro_mm
    return obj


def detectar_tablero(img: np.ndarray, patron: PatronCalibracion):
    """Busca las esquinas internas. Devuelve (encontrado, esquinas Nx1x2 float32)."""
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, esq = cv2.findChessboardCornersSB(
        gris, patron.esquinas, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if ok:
        return True, esq.astype(np.float32)
    ok, esq = cv2.findChessboardCorners(
        gris, patron.esquinas,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return False, None
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    esq = cv2.cornerSubPix(gris, esq, (11, 11), (-1, -1), criterio)
    return True, esq.astype(np.float32)


# ---------------------------------------------------------------------------
# Calibración individual
# ---------------------------------------------------------------------------
@dataclass
class CalibracionCamara:
    K: np.ndarray
    dist: np.ndarray
    tam_imagen: tuple            # (ancho, alto)
    rms: float                   # error de reproyección RMS (px)
    errores_vista: list = field(default_factory=list)
    vistas: list = field(default_factory=list)   # nombres de archivo usados

    def guardar(self, ruta: Path) -> Path:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ruta, K=self.K, dist=self.dist, tam_imagen=np.array(self.tam_imagen),
                 rms=self.rms, errores_vista=np.array(self.errores_vista),
                 vistas=np.array(self.vistas))
        return ruta

    @classmethod
    def cargar(cls, ruta: Path) -> "CalibracionCamara":
        d = np.load(ruta, allow_pickle=False)
        return cls(d["K"], d["dist"], tuple(int(v) for v in d["tam_imagen"]),
                   float(d["rms"]), d["errores_vista"].tolist(), d["vistas"].tolist())

    def resumen(self) -> str:
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        return (f"Imagen {self.tam_imagen[0]}x{self.tam_imagen[1]} | vistas: {len(self.vistas)} | "
                f"RMS = {self.rms:.3f} px\n  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}\n"
                f"  distorsión = {np.round(self.dist.ravel(), 4)}")


def escalar_calibracion(cal: CalibracionCamara, tam: tuple) -> CalibracionCamara:
    """Adapta K a otra resolución con la MISMA relación de aspecto.

    fx, fy, cx, cy escalan linealmente con el tamaño de la imagen; los
    coeficientes de distorsión no cambian (están en coordenadas normalizadas).
    """
    w0, h0 = cal.tam_imagen
    w1, h1 = tam
    if abs(w0 / h0 - w1 / h1) > 0.01:
        raise ValueError(f"Relación de aspecto distinta: {cal.tam_imagen} vs {tam}. "
                         "Recalibre con fotos de la misma resolución/orientación.")
    sx, sy = w1 / w0, h1 / h0
    K = cal.K.copy()
    K[0, 0] *= sx
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1] *= sy
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return CalibracionCamara(K, cal.dist.copy(), tuple(tam), cal.rms,
                             cal.errores_vista, cal.vistas)


DIAGONAL_35MM = 43.267   # diagonal del fotograma de 35 mm (36 x 24 mm)


def f35_desde_exif(ruta: Path) -> float | None:
    """Focal equivalente a 35 mm guardada por el celular en el EXIF, o None.

    WhatsApp y otras apps de mensajería borran el EXIF: la foto debe pasarse
    por cable, AirDrop, iCloud o correo como archivo original.
    """
    from PIL import Image

    with Image.open(ruta) as im:
        valor = im.getexif().get_ifd(0x8769).get(0xA405)   # FocalLengthIn35mmFilm
    return float(valor) if valor else None


def calibracion_aproximada(tam: tuple, f35_mm: float) -> CalibracionCamara:
    """K aproximada SIN tablero, a partir de la focal equivalente a 35 mm.

    La focal "equivalente" compara el ángulo de visión con el de una cámara de
    35 mm (diagonal 43.27 mm): f_px = f35 · diagonal_px / 43.27. Supone píxeles
    cuadrados, centro óptico en el centro de la imagen y sin distorsión (los
    celulares ya corrigen la del lente principal). Es menos exacta que calibrar
    con tablero (error típico de unos pocos %) y no se puede verificar con un RMS.
    """
    ancho, alto = tam
    f = f35_mm * float(np.hypot(ancho, alto)) / DIAGONAL_35MM
    K = np.array([[f, 0, (ancho - 1) / 2], [0, f, (alto - 1) / 2], [0, 0, 1]], np.float64)
    return CalibracionCamara(K, np.zeros((1, 5)), (int(ancho), int(alto)), float("nan"),
                             [], [f"aproximada (f35 = {f35_mm:g} mm)"])


def calibrar_desde_puntos(obj_pts: list, img_pts: list, tam: tuple,
                          vistas: list | None = None) -> CalibracionCamara:
    """cv2.calibrateCamera a partir de correspondencias 3D-2D ya detectadas."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, tam, None, None)
    errores = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proy, _ = cv2.projectPoints(o, r, t, K, dist)
        # Ambos a (N, 2): findChessboardCornersSB da (N, 2) y projectPoints (N, 1, 2);
        # restarlos sin más los combina en una matriz N x N y el error sale enorme.
        dif = proy.reshape(-1, 2) - np.asarray(i).reshape(-1, 2)
        errores.append(float(np.sqrt(np.mean(np.sum(dif ** 2, axis=1)))))
    return CalibracionCamara(K, dist, tuple(tam), float(rms), errores,
                             vistas or [str(n) for n in range(len(obj_pts))])


def calibrar_camara(rutas: list[Path], patron: PatronCalibracion,
                    tam: tuple | None = None, min_vistas: int = 10,
                    verbose: bool = True) -> CalibracionCamara:
    """Calibración intrínseca (K, distorsión) con fotos del tablero.

    ``tam=(ancho, alto)``: si se indica, cada foto se armoniza a ese tamaño
    (debe coincidir con el usado luego para las escenas).
    """
    obj = puntos_objeto(patron)
    obj_pts, img_pts, usadas = [], [], []
    tam_real = None
    for ruta in rutas:
        img = cargar_imagen(ruta)
        if tam is not None:
            img = armonizar(img, tam)
        t = (img.shape[1], img.shape[0])
        if tam_real is None:
            tam_real = t
        elif t != tam_real:
            if verbose:
                print(f"  [omitida] {ruta.name}: tamaño {t} distinto de {tam_real}")
            continue
        ok, esq = detectar_tablero(img, patron)
        if verbose:
            print(f"  {'[ok]     ' if ok else '[sin patrón]'} {ruta.name}")
        if ok:
            obj_pts.append(obj)
            img_pts.append(esq)
            usadas.append(ruta.name)
    if len(obj_pts) < min_vistas:
        raise RuntimeError(f"Solo {len(obj_pts)} vistas válidas del patrón; se requieren "
                           f">= {min_vistas}. Tome más fotos variando ángulo y posición.")
    return calibrar_desde_puntos(obj_pts, img_pts, tam_real, usadas)


# ---------------------------------------------------------------------------
# Calibración estéreo
# ---------------------------------------------------------------------------
@dataclass
class CalibracionEstereo:
    K1: np.ndarray
    d1: np.ndarray
    K2: np.ndarray
    d2: np.ndarray
    R: np.ndarray
    T: np.ndarray                # mm
    tam_imagen: tuple
    rms: float | None = None     # None si la pose no proviene de stereoCalibrate
    metodo: str = "stereoCalibrate"
    E: np.ndarray | None = None
    F: np.ndarray | None = None

    @property
    def linea_base_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    def guardar(self, ruta: Path) -> Path:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        extra = {k: v for k, v in (("E", self.E), ("F", self.F)) if v is not None}
        np.savez(ruta, K1=self.K1, d1=self.d1, K2=self.K2, d2=self.d2, R=self.R, T=self.T,
                 tam_imagen=np.array(self.tam_imagen),
                 rms=np.nan if self.rms is None else self.rms,
                 metodo=np.array(self.metodo), **extra)
        return ruta

    @classmethod
    def cargar(cls, ruta: Path) -> "CalibracionEstereo":
        d = np.load(ruta, allow_pickle=False)
        rms = float(d["rms"])
        return cls(d["K1"], d["d1"], d["K2"], d["d2"], d["R"], d["T"],
                   tuple(int(v) for v in d["tam_imagen"]),
                   None if np.isnan(rms) else rms, str(d["metodo"]),
                   d["E"] if "E" in d else None, d["F"] if "F" in d else None)

    def resumen(self) -> str:
        ang = np.degrees(np.linalg.norm(cv2.Rodrigues(self.R)[0]))
        rms = "n/a" if self.rms is None else f"{self.rms:.3f} px"
        return (f"Método: {self.metodo} | RMS estéreo: {rms}\n"
                f"  Línea base |T| = {self.linea_base_mm:.1f} mm  T = {np.round(self.T.ravel(), 1)}\n"
                f"  Rotación relativa = {ang:.2f}°")


def detectar_pares_tablero(rutas_izq: list[Path], rutas_der: list[Path],
                           patron: PatronCalibracion, tam: tuple, verbose=True):
    """Detecta el patrón en pares (se usan solo los pares donde aparece en ambas)."""
    obj = puntos_objeto(patron)
    obj_pts, pts_i, pts_d = [], [], []
    for ri, rd in zip(rutas_izq, rutas_der):
        ii = armonizar(cargar_imagen(ri), tam)
        idr = armonizar(cargar_imagen(rd), tam)
        ok_i, e_i = detectar_tablero(ii, patron)
        ok_d, e_d = detectar_tablero(idr, patron)
        if verbose:
            print(f"  {ri.name} / {rd.name}: izq={'ok' if ok_i else '--'} der={'ok' if ok_d else '--'}")
        if ok_i and ok_d:
            obj_pts.append(obj)
            pts_i.append(e_i)
            pts_d.append(e_d)
    return obj_pts, pts_i, pts_d


def calibrar_estereo_desde_puntos(obj_pts, pts_i, pts_d, cal_i: CalibracionCamara,
                                  cal_d: CalibracionCamara,
                                  refinar_intrinsecos: bool = False) -> CalibracionEstereo:
    """cv2.stereoCalibrate. Por defecto fija los intrínsecos de cada cámara
    (calibradas por separado), lo que es más estable con cámaras distintas."""
    if cal_i.tam_imagen != cal_d.tam_imagen:
        raise ValueError("Las dos calibraciones deben tener el mismo tamaño de imagen "
                         f"({cal_i.tam_imagen} vs {cal_d.tam_imagen}). Use armonizar().")
    flags = cv2.CALIB_USE_INTRINSIC_GUESS if refinar_intrinsecos else cv2.CALIB_FIX_INTRINSIC
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, pts_i, pts_d, cal_i.K.copy(), cal_i.dist.copy(),
        cal_d.K.copy(), cal_d.dist.copy(), cal_i.tam_imagen,
        criteria=criterio, flags=flags)
    return CalibracionEstereo(K1, d1, K2, d2, R, T, cal_i.tam_imagen, float(rms),
                              "stereoCalibrate", E, F)


# ---------------------------------------------------------------------------
# Modo 2: pose relativa entre dos fotos del mismo celular
# ---------------------------------------------------------------------------
def pose_por_tablero(img_izq, img_der, cal: CalibracionCamara,
                     patron: PatronCalibracion) -> CalibracionEstereo:
    """Pose relativa métrica usando un tablero visible en AMBAS fotos.

    solvePnP da la pose del tablero en cada vista (R_i, t_i). La transformación
    izquierda -> derecha es R = R_d R_i^T, T = t_d - R t_i (en mm).
    """
    obj = puntos_objeto(patron)
    poses = []
    for img in (img_izq, img_der):
        ok, esq = detectar_tablero(img, patron)
        if not ok:
            raise RuntimeError("No se detectó el tablero en ambas imágenes.")
        ok, rvec, tvec = cv2.solvePnP(obj, esq, cal.K, cal.dist)
        poses.append((cv2.Rodrigues(rvec)[0], tvec))
    (Ri, ti), (Rd, td) = poses
    R = Rd @ Ri.T
    T = td - R @ ti
    return CalibracionEstereo(cal.K, cal.dist, cal.K, cal.dist, R, T, cal.tam_imagen,
                              None, "solvePnP (tablero en ambas vistas)")


def emparejar_caracteristicas(img_izq, img_der, max_pts: int = 4000, ratio: float = 0.75):
    """SIFT + prueba de razón de Lowe. Devuelve puntos (N,2) izquierda y derecha."""
    sift = cv2.SIFT_create(nfeatures=max_pts)
    gi = cv2.cvtColor(img_izq, cv2.COLOR_BGR2GRAY)
    gd = cv2.cvtColor(img_der, cv2.COLOR_BGR2GRAY)
    kp_i, des_i = sift.detectAndCompute(gi, None)
    kp_d, des_d = sift.detectAndCompute(gd, None)
    if des_i is None or des_d is None or len(kp_i) < 8 or len(kp_d) < 8:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    # Búsqueda exacta (no FLANN): FLANN es aleatorio y el mismo par daría
    # correspondencias, y por tanto poses, distintas en cada ejecución.
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    buenos = [m for m, n in (p for p in matcher.knnMatch(des_i, des_d, k=2) if len(p) == 2)
              if m.distance < ratio * n.distance]
    pi = np.float32([kp_i[m.queryIdx].pt for m in buenos])
    pd_ = np.float32([kp_d[m.trainIdx].pt for m in buenos])
    return pi, pd_


def pose_por_esencial(img_izq, img_der, cal: CalibracionCamara,
                      linea_base_mm: float) -> tuple[CalibracionEstereo, dict]:
    """Pose relativa con la matriz esencial (K conocida) + línea base MEDIDA.

    La matriz esencial solo da la dirección de la traslación (escala
    desconocida). La escala métrica se obtiene de la distancia que se desplazó
    el celular, medida con una regla. Un error en esa medida produce un error
    proporcional en la profundidad.
    """
    pi, pd_ = emparejar_caracteristicas(img_izq, img_der)
    if len(pi) < 15:
        raise RuntimeError(f"Muy pocas correspondencias ({len(pi)}).")
    ni = cv2.undistortPoints(pi.reshape(-1, 1, 2), cal.K, cal.dist)
    nd = cv2.undistortPoints(pd_.reshape(-1, 1, 2), cal.K, cal.dist)
    # RANSAC descarta los emparejamientos malos; LMEDS refina con los buenos. Solo
    # con RANSAC la rotación variaba hasta ~0.5°, y medio grado mueve la
    # profundidad ~20 % con una línea base de 6 cm.
    umbral = 1.0 / cal.K[0, 0]                       # 1 px en coordenadas normalizadas
    _, inl = cv2.findEssentialMat(ni, nd, np.eye(3), cv2.RANSAC, 0.999, umbral)
    buenos = inl.ravel() > 0
    ni, nd = ni[buenos], nd[buenos]
    E, inl = cv2.findEssentialMat(ni, nd, np.eye(3), cv2.LMEDS, 0.999, umbral)
    _, R, t, inl_pose = cv2.recoverPose(E, ni, nd, np.eye(3), mask=inl)
    T = t / np.linalg.norm(t) * linea_base_mm
    info = {"correspondencias": len(pi), "inliers": int((inl_pose > 0).sum())}
    return CalibracionEstereo(cal.K, cal.dist, cal.K, cal.dist, R, T, cal.tam_imagen,
                              None, "matriz esencial + línea base medida", E), info


# ---------------------------------------------------------------------------
# Rectificación
# ---------------------------------------------------------------------------
@dataclass
class Rectificacion:
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    mapas_izq: tuple
    mapas_der: tuple
    roi_izq: tuple
    roi_der: tuple

    @property
    def f_px(self) -> float:
        return float(self.P1[0, 0])


def calcular_rectificacion(cal: CalibracionEstereo, alpha: float = 0.0) -> Rectificacion:
    """cv2.stereoRectify + cv2.initUndistortRectifyMap.

    Los mapas corrigen la distorsión de lente y alinean las filas de ambas
    imágenes (líneas epipolares horizontales). alpha=0 recorta a la zona válida.
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cal.K1, cal.d1, cal.K2, cal.d2, cal.tam_imagen, cal.R, cal.T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    m_i = cv2.initUndistortRectifyMap(cal.K1, cal.d1, R1, P1, cal.tam_imagen, cv2.CV_32FC1)
    m_d = cv2.initUndistortRectifyMap(cal.K2, cal.d2, R2, P2, cal.tam_imagen, cv2.CV_32FC1)
    return Rectificacion(R1, R2, P1, P2, Q, m_i, m_d, roi1, roi2)


def rectificar_par(img_izq, img_der, rect: Rectificacion):
    ri = cv2.remap(img_izq, *rect.mapas_izq, cv2.INTER_LINEAR)
    rd = cv2.remap(img_der, *rect.mapas_der, cv2.INTER_LINEAR)
    return ri, rd


def rectificacion_no_calibrada(img_izq, img_der):
    """Rectificación sin calibración (cv2.stereoRectifyUncalibrated).

    Solo permite disparidad RELATIVA: sin K ni línea base no hay profundidad
    métrica. Útil para revisar si un par del celular es aprovechable.
    """
    pi, pd_ = emparejar_caracteristicas(img_izq, img_der)
    if len(pi) < 15:
        raise RuntimeError(f"Muy pocas correspondencias ({len(pi)}).")
    F, mask = cv2.findFundamentalMat(pi, pd_, cv2.FM_RANSAC, 1.0, 0.999)
    inl = mask.ravel() == 1
    h, w = img_izq.shape[:2]
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pi[inl], pd_[inl], F, (w, h))
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated falló.")
    ri = cv2.warpPerspective(img_izq, H1, (w, h))
    rd = cv2.warpPerspective(img_der, H2, (w, h))
    return ri, rd, {"F": F, "H1": H1, "H2": H2, "correspondencias": len(pi),
                    "inliers": int(inl.sum())}


def error_epipolar_vertical(rect_izq, rect_der) -> dict:
    """Calidad de la rectificación: |y_izq - y_der| de correspondencias SIFT.

    En un par bien rectificado los puntos homólogos están en la misma fila, así
    que la mediana debería ser cercana a 0 (típicamente < 1-2 px).
    """
    pi, pd_ = emparejar_caracteristicas(rect_izq, rect_der)
    if len(pi) < 8:
        return {"n": len(pi), "mediana_px": np.nan, "p90_px": np.nan}
    dy = np.abs(pi[:, 1] - pd_[:, 1])
    return {"n": len(pi), "mediana_px": float(np.median(dy)),
            "p90_px": float(np.percentile(dy, 90))}


# ---------------------------------------------------------------------------
# Disparidad y profundidad
# ---------------------------------------------------------------------------
def crear_sgbm(p: ParametrosSGBM) -> cv2.StereoSGBM:
    canales = 3
    modos = {"SGBM": cv2.STEREO_SGBM_MODE_SGBM, "HH": cv2.STEREO_SGBM_MODE_HH,
             "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY}
    return cv2.StereoSGBM_create(
        minDisparity=p.min_disparidad, numDisparities=p.num_disparidades,
        blockSize=p.tam_bloque,
        P1=8 * canales * p.tam_bloque ** 2, P2=32 * canales * p.tam_bloque ** 2,
        disp12MaxDiff=p.disp12_max_diff, uniquenessRatio=p.uniqueness,
        speckleWindowSize=p.speckle_ventana, speckleRange=p.speckle_rango,
        mode=modos[p.modo])


def calcular_disparidad(rect_izq, rect_der, p: ParametrosSGBM | None = None) -> np.ndarray:
    """Mapa de disparidad (px, float32) con StereoSGBM. Inválidos = NaN.

    SGBM devuelve enteros con 4 bits fraccionarios (valor real * 16).
    """
    p = p or ParametrosSGBM()
    disp = crear_sgbm(p).compute(rect_izq, rect_der).astype(np.float32) / 16.0
    disp[disp <= p.min_disparidad - 1 + 1e-6] = np.nan
    disp[disp <= 0] = np.nan
    return disp


def nube_3d(disp: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """cv2.reprojectImageTo3D: (X, Y, Z) en mm para cada píxel. Z = f*B/d."""
    d = np.nan_to_num(disp, nan=0.0).astype(np.float32)
    pts = cv2.reprojectImageTo3D(d, Q, handleMissingValues=True)
    pts[~np.isfinite(disp)] = np.nan
    pts[np.abs(pts[..., 2]) > 1e5] = np.nan
    return pts


def medir_tomates(df: pd.DataFrame, detecciones: list, escala: float,
                  puntos3d: np.ndarray, f_px: float,
                  min_validos: float = 0.2) -> pd.DataFrame:
    """Profundidad y dimensiones físicas de cada tomate detectado.

    Requiere que la detección se haya hecho sobre la imagen IZQUIERDA
    RECTIFICADA (mismas coordenadas que el mapa de disparidad).

    * Profundidad Z: mediana de Z dentro de la máscara erosionada del fruto
      (robusta a bordes y píxeles sin disparidad).
    * Dimensiones (modelo pinhole): tamaño_mm = tamaño_px * Z / f.
      Válido si el fruto es aproximadamente frontal a la cámara.
    """
    filas = []
    alto, ancho = puntos3d.shape[:2]
    for d in detecciones:
        x, y, w, h = d.bbox
        m = cv2.erode(d.mascara_local, np.ones((5, 5), np.uint8))
        # Coordenadas en la imagen rectificada a resolución completa.
        x0, y0 = int(x / escala), int(y / escala)
        x1, y1 = min(int((x + w) / escala), ancho), min(int((y + h) / escala), alto)
        m_full = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        Z = puntos3d[y0:y1, x0:x1, 2][m_full]
        validos = np.isfinite(Z) & (Z > 0)
        frac = validos.mean() if Z.size else 0.0
        fila = {"id": d.id, "frac_disparidad_valida": frac}
        if frac >= min_validos:
            z = float(np.median(Z[validos]))
            fr = df.loc[df["id"] == d.id].iloc[0]
            fila.update({
                "profundidad_mm": z,
                "ancho_mm": fr["ancho_bbox_px_orig"] * z / f_px,
                "alto_mm": fr["alto_bbox_px_orig"] * z / f_px,
                "diametro_equiv_mm": fr["diametro_equiv_px_orig"] * z / f_px,
            })
        else:
            fila.update({"profundidad_mm": np.nan, "ancho_mm": np.nan,
                         "alto_mm": np.nan, "diametro_equiv_mm": np.nan})
        filas.append(fila)
    return pd.DataFrame(filas)
