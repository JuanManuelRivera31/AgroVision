"""Reconocimiento automático de la fruta de la foto (tomate, banano, mango, manzana...).

Idea (visión clásica + SVM, igual que la madurez):

1. La foto se segmenta con el perfil de CADA fruta. Cada perfil encuentra
   "objetos candidatos" (frutos reales o, a veces, hojas y fondo).
2. De cada objeto se mide un vector de características que no depende de la
   fruta: color (tono, saturación, brillo, a*, b*, histograma de tono), forma
   (alargamiento, circularidad, solidez) y textura (desviación local).
3. Una SVM entrenada con fotos de referencia (``data/referencias_frutas.csv``:
   imagen y fruta) dice de qué fruta parece cada objeto.
4. Se quitan los objetos repetidos (el mismo fruto hallado por dos perfiles) y
   cada objeto vota por la fruta a la que se parece, con peso = su área.
   Gana la fruta con más área.

El modelo se guarda en ``modelos/reconocedor_frutas.joblib`` y se vuelve a
entrenar solo si cambian las referencias o los perfiles.

Para que reconozca mejor: agregar al CSV una o dos fotos de cada fruta tomadas
con la misma cámara (DroidCam) y en el mismo lugar donde se va a usar.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from . import config as cfg
from .procesamiento import cargar_imagen, desviacion_local, redimensionar
from .segmentacion import metricas_contorno, segmentar

ARCHIVO_REFERENCIAS = cfg.DATA / "referencias_frutas.csv"
ARCHIVO_MODELO = cfg.DIR_MODELOS / "reconocedor_frutas.joblib"
BINS_TONO = 9            # histograma de tono en 9 franjas de 20 (H de OpenCV: 0-179)


@dataclass
class Reconocimiento:
    fruta: str
    puntajes: pd.DataFrame          # una fila por fruta: objetos que se le parecen y puntaje (0-1)
    segmentaciones: dict            # {fruta: ResultadoSegmentacion} (reutilizable)


# ---------------------------------------------------------------------------
# Características de un objeto
# ---------------------------------------------------------------------------
def caracteristicas_objeto(img: np.ndarray, det) -> dict:
    """Vector de color, forma y textura de una detección (no depende del perfil)."""
    x, y, w, h = det.bbox
    roi = img[y:y + h, x:x + w]
    m = det.mascara_local > 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    H, S, V = (hsv[..., i][m].astype(np.float32) for i in range(3))
    validos = (S > 40) & (V > 40)
    Hv = H[validos] if validos.any() else H
    ang = Hv * (2 * np.pi / 180)               # el tono es circular (el rojo está en 0 y en 180)
    hist = np.histogram(Hv, bins=BINS_TONO, range=(0, 180))[0] / max(len(Hv), 1)

    (_, _), (lado1, lado2), _ = cv2.minAreaRect(det.contorno)
    mc = metricas_contorno(det.contorno)
    textura = desviacion_local(gris, 9)[m]
    return {
        "tono_cos": float(np.cos(ang).mean()), "tono_sin": float(np.sin(ang).mean()),
        "s_media": float(S.mean()), "v_media": float(V.mean()), "v_std": float(V.std()),
        "a_media": float(lab[..., 1][m].mean()), "b_media": float(lab[..., 2][m].mean()),
        **{f"hist_{i}": float(v) for i, v in enumerate(hist)},
        "alargamiento": float(max(lado1, lado2) / max(min(lado1, lado2), 1)),
        "circularidad": float(mc["circularidad"]), "solidez": float(mc["solidez"]),
        "textura": float(textura.mean()),
    }


def _cargar(ruta) -> np.ndarray:
    img, _ = redimensionar(cargar_imagen(ruta), 1280, ampliar=True)
    return img


def objetos_de_imagen(img: np.ndarray, perfiles: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Segmenta con cada perfil y mide cada objeto. Devuelve (tabla, segmentaciones)."""
    perfiles = perfiles or {f: cfg.cargar_perfil(f) for f in cfg.listar_perfiles()}
    filas, segs = [], {}
    for fruta, perfil in perfiles.items():
        res = segmentar(img, perfil.segmentacion)
        segs[fruta] = res
        for d in res.detecciones:
            filas.append({"perfil": fruta, "id": d.id, "area": float(cv2.contourArea(d.contorno)),
                          "cx": d.bbox[0] + d.bbox[2] / 2, **caracteristicas_objeto(res.imagen, d)})
    return pd.DataFrame(filas), segs


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def _firma() -> str:
    """Cambia si cambian las referencias, sus imágenes o los perfiles -> reentrenar."""
    h = hashlib.sha1()
    if ARCHIVO_REFERENCIAS.is_file():
        h.update(ARCHIVO_REFERENCIAS.read_bytes())
        for img in pd.read_csv(ARCHIVO_REFERENCIAS, encoding="utf-8-sig")["imagen"]:
            ruta = cfg.DIR_IMAGENES / img
            h.update(f"{img}:{ruta.stat().st_size if ruta.is_file() else 0}".encode())
    for p in sorted(cfg.DIR_PERFILES.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def tabla_entrenamiento() -> pd.DataFrame:
    """Objetos de las fotos de referencia, con la fruta de la foto como etiqueta.

    Se usan los objetos que encuentran TODOS los perfiles (así hay ejemplos de
    sobra de cada fruta), quitando los repetidos (mismo objeto hallado por dos perfiles).
    """
    refs = pd.read_csv(ARCHIVO_REFERENCIAS, encoding="utf-8-sig")
    perfiles = {f: cfg.cargar_perfil(f) for f in cfg.listar_perfiles()}
    tablas = []
    for _, r in refs.iterrows():
        ruta = cfg.DIR_IMAGENES / r["imagen"]
        if not ruta.is_file():
            print(f"AVISO: no está la referencia {r['imagen']}; se omite.")
            continue
        df, segs = objetos_de_imagen(_cargar(ruta), perfiles)
        if df.empty:
            continue
        df = _sin_repetidos(df, segs)
        df["imagen"], df["fruta"] = r["imagen"], r["fruta"]
        tablas.append(df)
    return pd.concat(tablas, ignore_index=True)


def _sin_repetidos(df: pd.DataFrame, segs: dict, iou_max: float = 0.5) -> pd.DataFrame:
    from .evaluacion import iou
    cajas = {(d_p, d.id): d.bbox for d_p, res in segs.items() for d in res.detecciones}
    conservar, vistas = [], []
    for i, f in df.sort_values("area", ascending=False).iterrows():
        caja = cajas[(f["perfil"], f["id"])]
        if all(iou(caja, v) < iou_max for v in vistas):
            conservar.append(i)
            vistas.append(caja)
    return df.loc[sorted(conservar)].reset_index(drop=True)


def columnas_caracteristicas(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("perfil", "id", "area", "cx", "imagen", "fruta")]


def nuevo_clasificador():
    """SVM (núcleo RBF) con clases balanceadas: hay muchas más fotos de tomate
    que de las otras frutas, y sin balancear casi todo saldría "tomate"."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced"))


def entrenar(guardar: bool = True):
    """Entrena la SVM con las referencias. Devuelve (modelo, tabla de entrenamiento)."""
    import joblib
    df = tabla_entrenamiento()
    cols = columnas_caracteristicas(df)
    modelo = nuevo_clasificador().fit(df[cols], df["fruta"])
    if guardar:
        cfg.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": modelo, "columnas": cols, "firma": _firma()}, ARCHIVO_MODELO)
    return modelo, df


def cargar_modelo():
    """El modelo guardado, o uno nuevo si cambió algo desde que se entrenó."""
    import joblib
    # joblib (pickle) solo carga el archivo que genera este mismo módulo en
    # modelos/ (igual que el clasificador de madurez); no cargar modelos ajenos.
    if ARCHIVO_MODELO.is_file():
        datos = joblib.load(ARCHIVO_MODELO)
        if datos.get("firma") == _firma():
            return datos["modelo"], datos["columnas"]
    print("Entrenando el reconocedor de frutas con data/referencias_frutas.csv ...")
    modelo, df = entrenar()
    return modelo, columnas_caracteristicas(df)


# ---------------------------------------------------------------------------
# Reconocimiento
# ---------------------------------------------------------------------------
def reconocer_fruta(img: np.ndarray, modelo=None, columnas=None) -> Reconocimiento:
    """Decide qué fruta hay en la foto. ``img`` en BGR (cualquier tamaño)."""
    if modelo is None:
        modelo, columnas = cargar_modelo()
    img, _ = redimensionar(img, 1280, ampliar=True)
    df, segs = objetos_de_imagen(img)
    frutas = list(segs)
    if df.empty:
        tabla = pd.DataFrame({"fruta": frutas, "objetos": 0, "puntaje": 0.0})
        return Reconocimiento(frutas[0], tabla, segs)
    df["parece"] = modelo.predict(df[columnas])
    # Cada fruta se puntúa con los objetos que SU PROPIO perfil encontró y que además
    # se le parecen: un trozo de fondo o de otra fruta que un perfil ajeno marca como
    # "fruto" no le suma a nadie. Peso = área (un fruto grande y claro pesa más).
    propios = df[df["perfil"] == df["parece"]]
    area = propios.groupby("parece")["area"].sum()
    votos = propios["parece"].value_counts()
    if area.sum() == 0:                       # ningún perfil se reconoce en su propio hallazgo
        area = df.groupby("parece")["area"].sum()
        votos = df["parece"].value_counts()
    tabla = pd.DataFrame({"fruta": frutas,
                          "objetos": [int(votos.get(f, 0)) for f in frutas],
                          "puntaje": [float(area.get(f, 0.0)) for f in frutas]})
    tabla["puntaje"] = (tabla["puntaje"] / max(tabla["puntaje"].sum(), 1.0)).round(3)
    tabla = tabla.sort_values("puntaje", ascending=False).reset_index(drop=True)
    return Reconocimiento(tabla.loc[0, "fruta"], tabla, segs)
