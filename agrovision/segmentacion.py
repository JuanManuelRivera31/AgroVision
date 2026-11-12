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


def redondez(c: np.ndarray) -> float:
    """Área del contorno / área del círculo mínimo que lo encierra (1 = círculo)."""
    _, r = cv2.minEnclosingCircle(c)
    return cv2.contourArea(c) / (np.pi * r * r) if r > 0 else 0.0


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

    # Sin marcador de fondo: si el fondo compitiera, se quedaría con el fruto que
    # no recibió semilla y ese fruto desaparecería. Watershed inunda todo el
    # recorte desde las semillas (1..n-1) y luego se recorta a la máscara.
    marcadores = cv2.watershed(img_bgr.copy(), marcadores.astype(np.int32))

    regiones = []
    for etiqueta in range(1, n):
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
    semillas = None
    if p.semillas_por_bordes:
        semillas, etapas["semillas"] = semillas_por_bordes(gris, distintiva, p, area_min)
    rechazadas: list[Rechazo] = []
    reg_distintivas = _dividir_candidatos(cand_distintivos, img, p, area_min, area_max,
                                          rechazadas, distintivos=True, semillas=semillas)
    reg_verdes = _dividir_candidatos(cand_verdes, img, p, area_min, area_max, rechazadas)

    # 6) Frutos a medio madurar: la parte verde y la parte distintiva salen como
    # regiones vecinas. Se unen si la unión es compacta (forma de un fruto).
    reg_distintivas, reg_verdes = _unir_pintones(reg_distintivas, reg_verdes, p, img.shape)

    # 7) Validación geométrica
    aceptadas: list[tuple] = []
    for cr, dividida in reg_distintivas + reg_verdes:
        mr, mask_r, frac_distintiva = _medir_region(cr, distintiva)
        if p.usar_envolvente and frac_distintiva >= 0.10:
            # Fruto de color distintivo tapado por tallos u otro fruto: se valida
            # (y se mide) con su envolvente convexa. Los verdes no: la envolvente
            # de un trozo de hoja también puede parecer redonda.
            # La máscara (de la que salen los colores para la madurez) se queda con
            # los píxeles visibles del fruto: la envolvente incluiría los tallos.
            visible = cr
            cr = cv2.convexHull(cr)
            mr = metricas_contorno(cr)
            rx, ry, rw, rh = mr["bbox"]
            mask_r = np.zeros((rh, rw), np.uint8)
            cv2.drawContours(mask_r, [_desplazar(visible, -rx, -ry)], -1, 255, cv2.FILLED)

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


def _medir_region(c: np.ndarray, distintiva: np.ndarray) -> tuple:
    """Métricas de forma, máscara local (bbox) y fracción de color distintivo."""
    mr = metricas_contorno(c)
    rx, ry, rw, rh = mr["bbox"]
    mask_r = np.zeros((rh, rw), np.uint8)
    cv2.drawContours(mask_r, [_desplazar(c, -rx, -ry)], -1, 255, cv2.FILLED)
    n_px = max(cv2.countNonZero(mask_r), 1)
    frac = cv2.countNonZero(cv2.bitwise_and(distintiva[ry:ry + rh, rx:rx + rw], mask_r)) / n_px
    return mr, mask_r, frac


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


def _todas_redondas(regiones: list[np.ndarray], umbral: float) -> bool:
    if len(regiones) < 2:
        return False
    for r in regiones:
        cs, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs or redondez(cv2.convexHull(max(cs, key=cv2.contourArea))) < umbral:
            return False
    return True


def semillas_por_bordes(gris: np.ndarray, distintiva: np.ndarray,
                        p: ParametrosSegmentacion, area_min: float) -> tuple:
    """Una semilla por fruto: máscara distintiva menos los bordes de Canny.

    Devuelve (etiquetas int32 con 0 = sin semilla, máscara binaria de semillas).
    """
    k = max(3, p.bordes_suavizado | 1)
    suave = cv2.GaussianBlur(gris, (k, k), 0)
    bordes = cv2.dilate(cv2.Canny(suave, *p.bordes_canny), kernel_eliptico(p.bordes_dilatacion))
    s = cv2.bitwise_and(distintiva, cv2.bitwise_not(bordes))
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel_eliptico(p.semillas_apertura))
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(s)
    pequenas = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < p.semillas_area_min * area_min)
    etiquetas[np.isin(etiquetas, pequenas[pequenas > 0])] = 0
    return etiquetas.astype(np.int32), ((etiquetas > 0) * 255).astype(np.uint8)


def _crecer_semillas(local: np.ndarray, img_bgr: np.ndarray, semillas: np.ndarray) -> list:
    """Watershed desde las semillas que caen dentro de la región ``local``."""
    etiquetas = np.unique(semillas[local > 0])
    etiquetas = etiquetas[etiquetas > 0]
    if len(etiquetas) < 2:
        return [local]
    marcadores = np.zeros(local.shape, np.int32)
    for i, e in enumerate(etiquetas, start=1):
        marcadores[(semillas == e) & (local > 0)] = i
    marcadores = cv2.watershed(img_bgr.copy(), marcadores)
    regiones = [((marcadores == i) & (local > 0)).astype(np.uint8) * 255
                for i in range(1, len(etiquetas) + 1)]
    return [r for r in regiones if cv2.countNonZero(r)] or [local]


def _solidez_aspecto(mascara: np.ndarray) -> tuple:
    cs, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0, 99.0
    m = metricas_contorno(max(cs, key=cv2.contourArea))
    return m["solidez"], m["aspecto"]


def _aspecto_rotado(mascara: np.ndarray) -> float:
    pts = cv2.findNonZero(mascara)
    if pts is None:
        return 1.0
    (_, _), (l1, l2), _ = cv2.minAreaRect(pts)
    return max(l1, l2) / max(min(l1, l2), 1.0)


def _sigue_alargado(a: np.ndarray, b: np.ndarray, union: np.ndarray) -> bool:
    """Dos frutos alargados (bananos) pegados lado a lado dan una unión MÁS ancha y
    menos alargada que cada uno; dos trozos de un mismo banano dan una unión más larga.
    Los trozos pequeños (astillas) siempre se unen al vecino."""
    area_a, area_b = cv2.countNonZero(a), cv2.countNonZero(b)
    if min(area_a, area_b) < 0.25 * max(area_a, area_b):
        return True
    mayor = max(_aspecto_rotado(a), _aspecto_rotado(b))
    return mayor < 2.5 or _aspecto_rotado(union) >= 0.85 * mayor


def _fusionar_fragmentos(regiones: list, p: ParametrosSegmentacion) -> list:
    """Vuelve a unir los trozos de un mismo fruto que las semillas de bordes partieron.

    Una fruta con manchas, brillos o textura tiene bordes internos y sale en
    varios trozos. Lo que distingue un trozo de un fruto vecino es el contacto:
    dos trozos del mismo fruto comparten una frontera tan larga como el trozo
    más pequeño es ancho; dos frutos pegados se tocan solo por una "cintura"
    estrecha. Se unen los vecinos cuya frontera mida al menos ``fusion_contacto``
    veces el diámetro equivalente del menor, siempre que la unión siga teniendo
    forma de fruto (``aspecto_max``).
    """
    regiones = list(regiones)
    k = np.ones((3, 3), np.uint8)
    while len(regiones) > 1:
        mejor = None
        for i in range(len(regiones)):
            for j in range(i + 1, len(regiones)):
                frontera = cv2.countNonZero(cv2.bitwise_and(cv2.dilate(regiones[i], k), regiones[j]))
                if frontera == 0:
                    continue                                   # no son vecinos
                menor = min(cv2.countNonZero(regiones[i]), cv2.countNonZero(regiones[j]))
                contacto = frontera / max(np.sqrt(4 * menor / np.pi), 1.0)
                if contacto >= p.fusion_contacto and (mejor is None or contacto > mejor[0]):
                    union = cv2.bitwise_or(regiones[i], regiones[j])
                    if _solidez_aspecto(union)[1] <= p.aspecto_max and                             _sigue_alargado(regiones[i], regiones[j], union):
                        mejor = (contacto, i, j, union)
        if mejor is None:
            break
        _, i, j, union = mejor
        regiones = [r for n, r in enumerate(regiones) if n not in (i, j)] + [union]
    return regiones


def _recrecer_nucleos(local: np.ndarray, img_bgr: np.ndarray, regiones: list,
                      kernel: np.ndarray) -> list:
    """Quita las colas finas de cada trozo (apertura) y deja que el núcleo crezca de
    nuevo por todo el candidato: así un trozo estrecho de un fruto no se pierde."""
    nucleos = [cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel) for r in regiones]
    nucleos = [n for n in nucleos if cv2.countNonZero(n)]
    if len(nucleos) < 2:
        return nucleos or regiones
    marcadores = np.zeros(local.shape, np.int32)
    for i, n in enumerate(nucleos, start=1):
        marcadores[n > 0] = i
    marcadores = cv2.watershed(img_bgr.copy(), marcadores)
    crecidos = [((marcadores == i) & (local > 0)).astype(np.uint8) * 255
                for i in range(1, len(nucleos) + 1)]
    return [r for r in crecidos if cv2.countNonZero(r)]


def _dividir_candidatos(contornos, img, p, area_min, area_max, rechazadas,
                        distintivos=False, semillas=None):
    """Aplica watershed a los candidatos poco sólidos. Devuelve [(contorno, dividida)].

    En candidatos ``distintivos`` con envolvente muy circular se asume un solo
    fruto tapado (tallos, sépalos) y no se divide. Si hay ``semillas`` (ver
    ``semillas_por_bordes``) y caen dos o más en un candidato, se divide con ellas.
    """
    salida = []
    for c in contornos:
        m = metricas_contorno(c)
        if m["area"] < area_min:
            continue  # ruido: no se lista como rechazo para no saturar la vista
        if m["area"] > 4 * area_max:
            rechazadas.append(Rechazo(c, m["bbox"], "área excesiva (fondo/follaje)"))
            continue
        x, y, w, h = m["bbox"]
        if semillas is not None:
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [_desplazar(c, -x, -y)], -1, 255, cv2.FILLED)
            regiones = _crecer_semillas(local, img[y:y + h, x:x + w], semillas[y:y + h, x:x + w])
            if len(regiones) > 1 and p.fusion_contacto > 0:
                regiones = _fusionar_fragmentos(regiones, p)
            if len(regiones) > 1:
                # Watershed deja colas finas (franjas de un fruto vecino que no
                # tenía semilla propia): una apertura las corta.
                k = kernel_eliptico(2 * p.semillas_apertura + 1)
                if p.fusion_contacto > 0:
                    regiones = _recrecer_nucleos(local, img[y:y + h, x:x + w], regiones, k)
                for r in regiones:
                    if p.fusion_contacto <= 0:
                        r = cv2.morphologyEx(r, cv2.MORPH_OPEN, k)
                    cs, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if cs:
                        salida.append((_desplazar(max(cs, key=cv2.contourArea), x, y), True))
                continue
        if not (p.usar_watershed and m["solidez"] < p.solidez_para_watershed):
            salida.append((c, False))
            continue
        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [_desplazar(c, -x, -y)], -1, 255, cv2.FILLED)
        regiones = separar_watershed(local, img[y:y + h, x:x + w],
                                     p.umbral_dist_watershed, p.ventana_picos_frac)
        # Varios frutos en corro también dan una envolvente redonda: un fruto tapado
        # se parte en cuñas, un racimo en regiones que son redondas por sí solas.
        if distintivos and p.redondez_envolvente_unico > 0 and \
                redondez(cv2.convexHull(c)) >= p.redondez_envolvente_unico and \
                not _todas_redondas(regiones, p.redondez_envolvente_unico):
            salida.append((c, False))
            continue
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
