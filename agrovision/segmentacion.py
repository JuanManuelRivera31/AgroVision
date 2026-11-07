"""Segmentación, separación (watershed) y validación geométrica de frutos.

Idea general (no se decide solo por color). Las clases de color las define el
perfil de la fruta y se agrupan en dos familias:

* distintivas: colores que no aparecen en el follaje (rojo, amarillo, café...).
* de follaje: colores que comparten fruto y hojas (normalmente el verde).

1. Color (HSV): una máscara por clase con ``cv2.inRange``.
2. Textura y bordes: la máscara de follaje se restringe a zonas lisas
   (desviación local baja, umbralizada con ``cv2.threshold``) y se corta por
   los bordes de Canny, porque las hojas tienen nervaduras y el fruto tiene
   contorno propio.
3. Morfología: apertura (quita ruido y estructuras finas como tallos/hojas
   delgadas) y cierre (une el fruto partido por reflejos).
4. Componentes con huecos rellenos (``cv2.findContours`` + ``cv2.drawContours``).
5. Regiones poco sólidas (posibles frutos pegados) -> watershed.
6. Unión de la parte verde y la distintiva de un mismo fruto a medio madurar.
7. Validación geométrica: área, circularidad, solidez y relación de aspecto.
   Los candidatos solo verdes (los que se confunden con hojas) deben cumplir
   criterios de forma más estrictos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import ParametrosSegmentacion
from .procesamiento import desviacion_local, preprocesar, redimensionar


@dataclass
class Deteccion:
    """Un fruto detectado. La máscara se guarda recortada a su bbox."""
    id: int
    contorno: np.ndarray          # contorno en coordenadas de la imagen procesada
    bbox: tuple                   # (x, y, w, h)
    mascara_local: np.ndarray     # uint8 0/255 de tamaño (h, w)
    fraccion_distintiva: float    # fracción de píxeles de color distintivo dentro del fruto
    por_watershed: bool = False

    def mascara_completa(self, forma: tuple) -> np.ndarray:
        m = np.zeros(forma[:2], np.uint8)
        x, y, w, h = self.bbox
        m[y:y + h, x:x + w] = self.mascara_local
        return m


@dataclass
class Rechazo:
    contorno: np.ndarray
    bbox: tuple
    motivo: str


@dataclass
class ResultadoSegmentacion:
    imagen: np.ndarray                     # BGR redimensionada (la que se procesó)
    escala: float                          # imagen = original * escala
    etapas: dict = field(default_factory=dict)
    detecciones: list = field(default_factory=list)
    rechazadas: list = field(default_factory=list)

    @property
    def conteo(self) -> int:
        return len(self.detecciones)


# ---------------------------------------------------------------------------
# Utilidades geométricas
# ---------------------------------------------------------------------------
def kernel_eliptico(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def ajuste_elipse(c: np.ndarray, bbox: tuple) -> float:
    """IoU entre la región y la elipse ajustada con cv2.fitEllipse (1 = elipse perfecta).

    Un fruto redondo (aunque esté algo ocluido) se parece a una elipse; los huecos de
    fondo entre hojas tienen formas irregulares.
    """
    if len(c) < 5:
        return 0.0
    x, y, w, h = bbox
    (cx, cy), ejes, ang = cv2.fitEllipse(c)
    m = max(int(max(ejes)), 2)
    forma = (h + 2 * m, w + 2 * m)
    region = np.zeros(forma, np.uint8)
    elipse = np.zeros(forma, np.uint8)
    cv2.drawContours(region, [(c - np.array([x - m, y - m], np.int32)).astype(np.int32)],
                     -1, 255, cv2.FILLED)
    cv2.ellipse(elipse, ((cx - x + m, cy - y + m), ejes, ang), 255, cv2.FILLED)
    inter = cv2.countNonZero(cv2.bitwise_and(region, elipse))
    union = cv2.countNonZero(cv2.bitwise_or(region, elipse))
    return inter / union if union else 0.0


def metricas_contorno(c: np.ndarray, con_elipse: bool = False) -> dict:
    """Métricas de forma con cv2.contourArea, cv2.arcLength y cv2.boundingRect.

    ``con_elipse`` agrega el ajuste a elipse (más costoso, se calcula solo cuando se usa).
    """
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    return {
        "ajuste_elipse": ajuste_elipse(c, (x, y, w, h)) if con_elipse else None,
        "area": area,
        "perimetro": perim,
        "circularidad": 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0,
        "solidez": area / hull_area if hull_area > 0 else 0.0,
        "aspecto": max(w, h) / max(min(w, h), 1),
        "bbox": (x, y, w, h),
    }


# ---------------------------------------------------------------------------
# Máscaras de color
# ---------------------------------------------------------------------------
def mascaras_color(hsv: np.ndarray, p: ParametrosSegmentacion) -> dict:
    """Una máscara por clase de color del perfil (unión de sus intervalos HSV)."""
    mascaras = {}
    for nombre, rangos in p.colores.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for bajo, alto in rangos:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, bajo, alto))
        mascaras[nombre] = m
    return mascaras


def _unir(mascaras: list, forma: tuple) -> np.ndarray:
    total = np.zeros(forma[:2], np.uint8)
    for m in mascaras:
        total = cv2.bitwise_or(total, m)
    return total


def mascara_lisa(gris: np.ndarray, p: ParametrosSegmentacion) -> tuple:
    """Zonas de textura baja mediante cv2.threshold sobre la desviación local."""
    sigma = desviacion_local(gris, p.textura_ventana)
    sigma8 = np.clip(sigma, 0, 255).astype(np.uint8)
    _, lisa = cv2.threshold(sigma8, p.textura_max_verde, 255,
                            cv2.THRESH_BINARY_INV)
    return lisa, sigma


# ---------------------------------------------------------------------------
# Watershed
# ---------------------------------------------------------------------------
def separar_watershed(mascara: np.ndarray, img_bgr: np.ndarray,
                      umbral_rel: float, ventana_frac: float = 0.6) -> list[np.ndarray]:
    """Separa frutos unidos usando la transformada de distancia + watershed.

    Las semillas son máximos locales de la distancia al borde (centros de los
    frutos). Si solo hay una semilla, la región se devuelve sin dividir.
    Trabaja sobre el recorte de la región para ser eficiente.
    """
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax < 3:
        return [mascara]
    # Máximo local: igual al máximo en una vecindad de lado ventana_frac*dmax.
    # (kernel rectangular: OpenCV lo aplica de forma separable, mucho más rápido)
    # Ojo: dos frutos cuyos centros estén más cerca que la ventana colapsan en
    # una sola semilla, y entonces el racimo no se separa.
    k = max(3, int(ventana_frac * dmax) | 1)
    dil = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    picos = ((dist >= dil - 1e-3) & (dist > umbral_rel * dmax)).astype(np.uint8) * 255
    picos = cv2.dilate(picos, kernel_eliptico(3))
    n, marcadores = cv2.connectedComponents(picos)
    if n <= 2:  # fondo + una sola semilla
        return [mascara]

    marcadores = marcadores + 1                    # fondo -> 1, semillas -> 2..n
    desconocido = cv2.subtract(mascara, picos)
    marcadores[desconocido == 255] = 0             # lo que watershed debe decidir
    marcadores = cv2.watershed(img_bgr.copy(), marcadores.astype(np.int32))

    regiones = []
    for etiqueta in range(2, n + 1):
        r = ((marcadores == etiqueta) & (mascara > 0)).astype(np.uint8) * 255
        if cv2.countNonZero(r) > 0:
            regiones.append(r)
    return regiones or [mascara]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def segmentar(img_bgr: np.ndarray, p: ParametrosSegmentacion) -> ResultadoSegmentacion:
    """Detecta los frutos con los parámetros de un perfil (``perfil.segmentacion``)."""
    img, escala = redimensionar(img_bgr, p.lado_max)
    alto, ancho = img.shape[:2]
    area_img = alto * ancho

    pre = preprocesar(img)
    hsv = pre["hsv"]
    gris = cv2.cvtColor(pre["mediana"], cv2.COLOR_BGR2GRAY)

    # 1) Color
    mc = mascaras_color(hsv, p)
    distintiva = _unir([m for n, m in mc.items() if n not in p.colores_follaje], img.shape)
    follaje = _unir([mc[n] for n in p.colores_follaje], img.shape)

    # 2) Textura y bordes (solo se aplican al color de follaje). Un fruto verde
    # junto a hojas verdes tiene el mismo color: lo que lo separa es su borde.
    # Se quitan de la máscara las zonas texturizadas y los bordes de Canny.
    lisa, sigma = mascara_lisa(gris, p)
    bordes = cv2.Canny(gris, p.canny_bajo, p.canny_alto)
    bordes_d = cv2.dilate(bordes, np.ones((3, 3), np.uint8))
    follaje_lisa = cv2.bitwise_and(follaje, lisa)
    follaje_lisa = cv2.bitwise_and(follaje_lisa, cv2.bitwise_not(bordes_d))

    # 3) Morfología. Distintivos y verdes se procesan por separado: la máscara
    # verde contiene hojas/fondo y no debe fusionarse con los frutos maduros.
    k_ap, k_ci = kernel_eliptico(p.kernel_apertura), kernel_eliptico(p.kernel_cierre)
    distintiva_ap = cv2.morphologyEx(distintiva, cv2.MORPH_OPEN, k_ap)
    distintiva_ci = cv2.morphologyEx(distintiva_ap, cv2.MORPH_CLOSE, k_ci)
    verde_ap = cv2.morphologyEx(follaje_lisa, cv2.MORPH_OPEN,
                                kernel_eliptico(p.kernel_apertura_verde))
    # Sin cierre en lo verde: volvería a unir el fruto con las hojas vecinas.
    apertura = cv2.bitwise_or(distintiva_ap, verde_ap)
    cierre = cv2.bitwise_or(distintiva_ci, verde_ap)

    # 4) Componentes con huecos rellenos (RETR_CCOMP: un fruto verde rodeado
    # por fondo verde queda como componente propio, no como hueco del fondo).
    cand_distintivos = componentes_rellenas(distintiva_ci)
    # Cada componente verde se cierra POR SEPARADO: tapa muescas de reflejos
    # sin volver a unirla con las hojas vecinas.
    cand_verdes = [cerrar_componente(c, k_ci, img.shape)
                   for c in componentes_rellenas(verde_ap)]
    final = np.zeros_like(distintiva)
    for c in cand_distintivos + cand_verdes:
        cv2.drawContours(final, [c], -1, 255, cv2.FILLED)

    etapas = {
        "preprocesada": pre["suavizada"],
        "iluminacion": pre["iluminacion"],
        "mediana": pre["mediana"],
        "hsv": hsv,
        "lab": pre["lab"],
        **{f"mascara_{n}": m for n, m in mc.items()},
        "textura_sigma": sigma,
        "mascara_lisa": lisa,
        "bordes": bordes,
        "mascara_follaje": follaje,
        "follaje_lisa": follaje_lisa,
        "mascara_distintiva": distintiva,
        "apertura": apertura,
        "cierre": cierre,
        "mascara_final": final,
    }

    # 5) Watershed sobre cada candidato poco sólido
    area_min, area_max = p.area_min_frac * area_img, p.area_max_frac * area_img
    rechazadas: list[Rechazo] = []
    reg_distintivas = _dividir_candidatos(cand_distintivos, img, p, area_min, area_max, rechazadas)
    reg_verdes = _dividir_candidatos(cand_verdes, img, p, area_min, area_max, rechazadas)

    # 6) Frutos a medio madurar: la parte verde y la parte distintiva salen como
    # regiones vecinas. Se unen si la unión es compacta (forma de un fruto).
    reg_distintivas, reg_verdes = _unir_pintones(reg_distintivas, reg_verdes, p, img.shape)

    # 7) Validación geométrica
    aceptadas: list[tuple] = []
    for cr, dividida in reg_distintivas + reg_verdes:
        mr = metricas_contorno(cr)
        rx, ry, rw, rh = mr["bbox"]
        mask_r = np.zeros((rh, rw), np.uint8)
        cv2.drawContours(mask_r, [_desplazar(cr, -rx, -ry)], -1, 255, cv2.FILLED)
        n_px = max(cv2.countNonZero(mask_r), 1)
        frac_distintiva = cv2.countNonZero(
            cv2.bitwise_and(distintiva[ry:ry + rh, rx:rx + rw], mask_r)) / n_px

        motivo = _validar(cr, mr, frac_distintiva, area_min, area_max, p, dividida)
        if motivo:
            rechazadas.append(Rechazo(cr, mr["bbox"], motivo))
        else:
            aceptadas.append((cr, mr["bbox"], mask_r, frac_distintiva, dividida))

    # 8) Área mínima relativa al tamaño típico de los frutos de esta foto.
    if p.area_min_rel > 0 and aceptadas:
        areas = [cv2.contourArea(t[0]) for t in aceptadas]
        corte = p.area_min_rel * float(np.median(areas))
        rechazadas += [Rechazo(t[0], t[1], "pequeña frente a los demás")
                       for t, a in zip(aceptadas, areas) if a < corte]
        aceptadas = [t for t, a in zip(aceptadas, areas) if a >= corte]

    # Identificadores únicos en orden de lectura (filas de arriba a abajo).
    fila = max(1, alto // 12)
    aceptadas.sort(key=lambda t: (t[1][1] // fila, t[1][0]))
    detecciones = [Deteccion(i + 1, c, b, mk, fc, ws)
                   for i, (c, b, mk, fc, ws) in enumerate(aceptadas)]

    return ResultadoSegmentacion(img, escala, etapas, detecciones, rechazadas)


def _desplazar(c: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return (c + np.array([dx, dy], np.int32)).astype(np.int32)


def _recortar_a_lienzo(c: np.ndarray, forma: tuple) -> np.ndarray:
    """Limita las coordenadas de un contorno al interior de la imagen.

    Los cierres morfológicos de ``cerrar_componente`` y ``_unir_pintones``
    trabajan en un lienzo con relleno, así que el contorno resultante puede
    sobresalir de la imagen cuando el fruto está cortado por el marco. Sin
    recortarlo, su ``cv2.boundingRect`` apunta fuera del lienzo y los recortes
    de máscara dejan de tener el tamaño del bbox.
    """
    alto, ancho = forma[:2]
    c = c.copy()
    c[..., 0] = np.clip(c[..., 0], 0, ancho - 1)
    c[..., 1] = np.clip(c[..., 1], 0, alto - 1)
    return c


def componentes_rellenas(mascara: np.ndarray) -> list[np.ndarray]:
    """Contornos exteriores de cada componente (incluidos los que están dentro
    de un hueco de otra componente). Al dibujarlos rellenos se tapan los huecos."""
    contornos, jer = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if jer is None:
        return []
    return [c for c, h in zip(contornos, jer[0]) if h[3] == -1]


def cerrar_componente(c: np.ndarray, kernel: np.ndarray, forma: tuple) -> np.ndarray:
    """Cierre morfológico de una sola componente (en su propio lienzo).

    ``forma`` es la forma de la imagen: el cierre puede crecer hasta el tamaño
    del kernel y el contorno se recorta para no salirse de ella.
    """
    x, y, w, h = cv2.boundingRect(c)
    m = kernel.shape[0]
    lienzo = np.zeros((h + 2 * m, w + 2 * m), np.uint8)
    cv2.drawContours(lienzo, [_desplazar(c, m - x, m - y)], -1, 255, cv2.FILLED)
    lienzo = cv2.morphologyEx(lienzo, cv2.MORPH_CLOSE, kernel)
    cs, _ = cv2.findContours(lienzo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return c
    return _recortar_a_lienzo(
        _desplazar(max(cs, key=cv2.contourArea), x - m, y - m), forma)


def _dividir_candidatos(contornos, img, p, area_min, area_max, rechazadas):
    """Aplica watershed a los candidatos poco sólidos. Devuelve [(contorno, dividida)]."""
    salida = []
    for c in contornos:
        m = metricas_contorno(c)
        if m["area"] < area_min:
            continue  # ruido: no se lista como rechazo para no saturar la vista
        if m["area"] > 4 * area_max:
            rechazadas.append(Rechazo(c, m["bbox"], "área excesiva (fondo/follaje)"))
            continue
        x, y, w, h = m["bbox"]
        if not (p.usar_watershed and m["solidez"] < p.solidez_para_watershed):
            salida.append((c, False))
            continue
        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [_desplazar(c, -x, -y)], -1, 255, cv2.FILLED)
        regiones = separar_watershed(local, img[y:y + h, x:x + w],
                                     p.umbral_dist_watershed, p.ventana_picos_frac)
        dividida = len(regiones) > 1
        for r in regiones:
            cs, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                salida.append((_desplazar(max(cs, key=cv2.contourArea), x, y), dividida))
    return salida


def _unir_pintones(distintivas, verdes, p, forma, holgura: int = 7):
    """Une regiones verde + distintiva adyacentes cuando forman un solo fruto.

    Criterio: las regiones se tocan (tras dilatar ``holgura`` px), la unión
    supera la solidez mínima y (si ``union_mas_circular``) es más circular que
    la región distintiva sola.

    ``forma`` es la forma de la imagen: la unión se calcula en un lienzo con
    holgura y el contorno se recorta para no salirse de ella.
    """
    usadas = set()
    nuevas_distintivas = []
    for cc, div in distintivas:
        mc = metricas_contorno(cc)
        x, y, w, h = mc["bbox"]
        mejor = None
        for j, (cv_, _) in enumerate(verdes):
            if j in usadas:
                continue
            xv, yv, wv, hv = cv2.boundingRect(cv_)
            if xv > x + w + holgura or x > xv + wv + holgura or \
               yv > y + h + holgura or y > yv + hv + holgura:
                continue
            x0, y0 = min(x, xv) - holgura, min(y, yv) - holgura
            x1, y1 = max(x + w, xv + wv) + holgura, max(y + h, yv + hv) + holgura
            lienzo = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.drawContours(lienzo, [_desplazar(cc, -x0, -y0), _desplazar(cv_, -x0, -y0)],
                             -1, 255, cv2.FILLED)
            unido = cv2.morphologyEx(lienzo, cv2.MORPH_CLOSE, kernel_eliptico(2 * holgura + 1))
            cs, _ = cv2.findContours(unido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(cs) != 1:
                continue
            mu = metricas_contorno(cs[0])
            mas_circular = mu["circularidad"] > mc["circularidad"] or not p.union_mas_circular
            if mu["solidez"] >= p.solidez_min and mas_circular and mu["aspecto"] <= p.aspecto_max:
                if mejor is None or mu["circularidad"] > mejor[2]:
                    mejor = (j, _desplazar(cs[0], x0, y0), mu["circularidad"])
        if mejor:
            usadas.add(mejor[0])
            nuevas_distintivas.append((_recortar_a_lienzo(mejor[1], forma), div))
        else:
            nuevas_distintivas.append((cc, div))
    return nuevas_distintivas, [v for j, v in enumerate(verdes) if j not in usadas]


def _validar(c: np.ndarray, m: dict, frac_distintiva: float, area_min: float, area_max: float,
             p: ParametrosSegmentacion, dividida: bool) -> str:
    """Devuelve el motivo de rechazo o cadena vacía si es un fruto válido."""
    if m["area"] < area_min:
        return "área pequeña"
    if m["area"] > area_max:
        return "área excesiva"
    if m["aspecto"] > p.aspecto_max:
        return "alargado"
    solo_verde = frac_distintiva < 0.10
    circ_min = p.circularidad_min_verde if solo_verde else p.circularidad_min
    sol_min = p.solidez_min_verde if solo_verde else p.solidez_min
    # Las regiones cortadas por watershed tienen un borde recto: se toleran
    # valores de forma algo menores.
    if dividida:
        circ_min *= 0.85
        sol_min *= 0.95
    if m["circularidad"] < circ_min:
        return "poco circular" + (" (verde)" if solo_verde else "")
    if m["solidez"] < sol_min:
        return "poco sólido" + (" (verde)" if solo_verde else "")
    if solo_verde and ajuste_elipse(c, m["bbox"]) < p.ajuste_elipse_min_verde * (0.95 if dividida else 1):
        return "no elíptico (verde)"
    return ""
