"""Generadores de datos SINTÉTICOS usados solo para pruebas automáticas del código.

No se usan en el Notebook ni representan resultados del sistema: sirven para
comprobar que las funciones calculan lo que deben cuando la respuesta es
conocida de antemano (geometría y colores controlados).
"""
from __future__ import annotations

import cv2
import numpy as np

from agrovision.config import PatronCalibracion
from agrovision.estereo import generar_tablero


def escena_tomates(semilla: int = 0):
    """Escena 960x720 con hojas texturizadas y tomates de color conocido.

    Devuelve (imagen BGR, lista de (cx, cy, r, categoria)).
    """
    rng = np.random.default_rng(semilla)
    h, w = 720, 960
    img = np.full((h, w, 3), (40, 70, 45), np.uint8)
    # Hojas: elipses verdes con nervaduras oscuras.
    for _ in range(45):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        ejes = (int(rng.integers(40, 110)), int(rng.integers(12, 30)))
        ang = float(rng.uniform(0, 180))
        verde = (int(rng.integers(30, 70)), int(rng.integers(110, 170)), int(rng.integers(40, 90)))
        cv2.ellipse(img, c, ejes, ang, 0, 360, verde, -1)
        for k in range(-ejes[0] + 10, ejes[0] - 10, 14):
            t = np.radians(ang)
            p = (int(c[0] + k * np.cos(t)), int(c[1] + k * np.sin(t)))
            q = (int(p[0] - 14 * np.sin(t + 0.6)), int(p[1] + 14 * np.cos(t + 0.6)))
            cv2.line(img, p, q, (20, 60, 25), 2)
    ruido = rng.normal(0, 6, img.shape)
    img = np.clip(img + ruido, 0, 255).astype(np.uint8)

    tomates = [
        (170, 180, 55, "Maduro"), (420, 160, 50, "Verde"), (680, 190, 58, "Pintón"),
        (220, 470, 52, "Verde"), (520, 470, 48, "Maduro"),
        # Dos maduros que se tocan (prueba de watershed)
        (745, 500, 50, "Maduro"), (838, 510, 48, "Maduro"),
    ]
    colores = {"Maduro": (35, 35, 200), "Verde": (70, 190, 120), "Pintón": None}
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r, cat in tomates:
        dentro = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        sombreado = 1.0 - 0.35 * (np.hypot(xx - cx + r * 0.3, yy - cy + r * 0.3) / (1.6 * r))
        if cat == "Pintón":
            # Mitad izquierda roja-anaranjada, mitad derecha verde-amarillenta.
            base = np.where((xx < cx)[..., None], np.array((30, 90, 220)), np.array((60, 190, 150)))
        else:
            base = np.broadcast_to(np.array(colores[cat]), img.shape)
        col = np.clip(base * sombreado[..., None], 0, 255)
        img[dentro] = col[dentro].astype(np.uint8)
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(3, r // 8), (235, 235, 235), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, tomates


def escena_tomates_en_bordes(semilla: int = 0):
    """Escena con tomates pintones pegados a los cuatro bordes de la imagen.

    Un pintón tiene una parte cálida y una parte verde que el detector une; si
    el fruto está cortado por el marco, esa unión puede salirse del lienzo.
    Devuelve (imagen BGR, lista de (cx, cy, r)).
    """
    img, _ = escena_tomates(semilla)
    h, w = img.shape[:2]
    r = 54
    centros = [(140, h - 30), (w - 30, 300), (300, 24), (24, 520)]
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in centros:
        dentro = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        # Mitad cálida y mitad verde, como en el pintón de escena_tomates.
        base = np.where((xx < cx)[..., None], np.array((30, 90, 220)), np.array((60, 190, 150)))
        img[dentro] = base[dentro].astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0), [(cx, cy, r) for cx, cy in centros]


def poligono_banano(centro, radio, ang_ini, ang_fin, ancho, n=40):
    """Contorno de un banano: arco grueso que se adelgaza hacia las puntas (grados)."""
    t = np.linspace(0, 1, n)
    ang = np.radians(ang_ini + (ang_fin - ang_ini) * t)
    medio = 0.5 * ancho * (0.25 + 0.75 * np.sin(np.pi * t) ** 0.6)
    cx, cy = centro
    exterior = np.stack([cx + (radio + medio) * np.cos(ang), cy + (radio + medio) * np.sin(ang)], 1)
    interior = np.stack([cx + (radio - medio) * np.cos(ang), cy + (radio - medio) * np.sin(ang)], 1)
    return np.vstack([exterior, interior[::-1]]).round().astype(np.int32)


def escena_bananos(semilla: int = 0):
    """Escena 960x720: bananos sueltos sobre un mantel claro.

    Devuelve (imagen BGR, lista de (bbox, categoria)).
    """
    rng = np.random.default_rng(semilla)
    h, w = 720, 960
    img = np.full((h, w, 3), (200, 205, 210), np.uint8)
    img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)

    bananos = [  # (centro del arco, radio, ángulo inicial, ángulo final, ancho, categoría)
        ((250, -40), 230, 55, 125, 60, "Maduro"),
        ((700, -60), 240, 60, 120, 58, "Verde"),
        ((230, 260), 220, 50, 120, 56, "Pintón"),
        ((690, 250), 230, 58, 122, 60, "Sobremaduro"),
        ((470, 330), 240, 62, 118, 54, "Maduro"),
    ]
    amarillo, verde, cafe = np.array((40, 210, 235)), np.array((60, 190, 120)), (25, 55, 95)
    yy, xx = np.mgrid[0:h, 0:w]
    salida = []
    for centro, radio, a0, a1, ancho, cat in bananos:
        poli = poligono_banano(centro, radio, a0, a1, ancho)
        dentro = np.zeros((h, w), np.uint8)
        cv2.fillPoly(dentro, [poli], 255)
        x, y, bw, bh = cv2.boundingRect(poli)
        if cat == "Verde":
            base = np.broadcast_to(verde, img.shape)
        elif cat == "Pintón":
            base = np.where((xx < x + bw // 2)[..., None], verde, amarillo)
        else:
            base = np.broadcast_to(amarillo, img.shape)
        sombreado = 1.0 - 0.25 * (yy - y) / max(bh, 1)
        col = np.clip(base * sombreado[..., None], 0, 255).astype(np.uint8)
        img[dentro > 0] = col[dentro > 0]
        if cat == "Sobremaduro":
            ys, xs = np.nonzero(dentro)
            for i in rng.choice(len(xs), 45, replace=False):
                cv2.circle(img, (int(xs[i]), int(ys[i])), int(rng.integers(4, 8)), cafe, -1)
        salida.append(((x, y, bw, bh), cat))
    # Las manchas no deben salirse del banano: se repinta el mantel fuera de ellos.
    mascara = np.zeros((h, w), np.uint8)
    for centro, radio, a0, a1, ancho, _ in bananos:
        cv2.fillPoly(mascara, [poligono_banano(centro, radio, a0, a1, ancho)], 255)
    fondo = np.clip(np.full((h, w, 3), (200, 205, 210)) + rng.normal(0, 5, (h, w, 3)), 0, 255)
    img[mascara == 0] = fondo[mascara == 0].astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0), salida


def escena_con_cajas(fruta: str):
    """(imagen, cajas reales) de la escena sintética de una fruta."""
    if fruta == "tomate":
        img, tomates = escena_tomates()
        return img, [(cx - r, cy - r, 2 * r, 2 * r) for cx, cy, r, _ in tomates]
    if fruta == "banano":
        img, bananos = escena_bananos()
        return img, [b for b, _ in bananos]
    raise ValueError(f"No hay escena sintética para '{fruta}'")


def camara_virtual(ancho=1280, alto=960, f=1100.0):
    K = np.array([[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]], np.float64)
    return K, (ancho, alto)


def render_tablero(K, tam, rvec, tvec, patron: PatronCalibracion, px_cuadro=60):
    """Renderiza el tablero visto por una cámara pinhole (sin distorsión).

    Se usa la homografía plano->imagen H = K [r1 r2 t] compuesta con la escala
    píxel_tablero -> mm.
    """
    margen = 60
    tablero = generar_tablero(patron, px_cuadro, margen)
    s = patron.tam_cuadro_mm / px_cuadro
    # Píxel del tablero -> coordenadas del plano (mm), con la primera esquina
    # interna en el origen (coincide con puntos_objeto()).
    A = np.array([[s, 0, -(margen + px_cuadro) * s], [0, s, -(margen + px_cuadro) * s], [0, 0, 1]])
    R = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    H = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, np.float64).ravel()]) @ A
    img = cv2.warpPerspective(tablero, H, tam, borderValue=180)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def poses_tablero(n=14, semilla=1):
    rng = np.random.default_rng(semilla)
    poses = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-160, 20), rng.uniform(-110, 0), rng.uniform(550, 800)])
        poses.append((rvec, tvec))
    return poses


def render_textura_plana(textura, mm_por_px, K, tam, rvec, tvec):
    """Renderiza una textura plana (Z=0 en su sistema, origen en su píxel (0,0)) vista por la cámara."""
    A = np.diag([mm_por_px, mm_por_px, 1.0])
    R = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    H = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, np.float64).ravel()]) @ A
    return cv2.warpPerspective(textura, H, tam, flags=cv2.INTER_LINEAR, borderValue=(90, 90, 90))


def lamina_tomates_y_tablero(patron: PatronCalibracion, mm_por_px=0.5):
    """Lámina plana: escena de tomates arriba y tablero abajo (mismo plano).

    El tablero se dibuja a la escala de la lámina para que su cuadro mida
    ``patron.tam_cuadro_mm``. Devuelve (textura, tomates, origen_tablero_px):
    origen_tablero_px es la posición de la primera esquina interna.
    """
    escena, tomates = escena_tomates()
    px_cuadro = int(round(patron.tam_cuadro_mm / mm_por_px))
    margen = 40
    tablero = cv2.cvtColor(generar_tablero(patron, px_cuadro, margen), cv2.COLOR_GRAY2BGR)
    franja = np.full((tablero.shape[0] + 20, escena.shape[1], 3), 255, np.uint8)
    x0 = (escena.shape[1] - tablero.shape[1]) // 2
    franja[10:10 + tablero.shape[0], x0:x0 + tablero.shape[1]] = tablero
    textura = np.vstack([escena, franja])
    origen = (x0 + margen + px_cuadro, escena.shape[0] + 10 + margen + px_cuadro)
    return textura, tomates, origen


def par_estereo_lamina(patron: PatronCalibracion, K, tam, B=60.0, Z=800.0, mm_por_px=0.5):
    """Par izquierda/derecha de la lámina vista por dos cámaras paralelas separadas B mm."""
    textura, tomates, _ = lamina_tomates_y_tablero(patron, mm_por_px)
    alto_mm, ancho_mm = textura.shape[0] * mm_por_px, textura.shape[1] * mm_por_px
    rvec = np.array([0.04, -0.05, 0.0])
    t_i = np.array([-ancho_mm / 2, -alto_mm / 2, Z])
    t_d = t_i + np.array([-B, 0.0, 0.0])
    izq = render_textura_plana(textura, mm_por_px, K, tam, rvec, t_i)
    der = render_textura_plana(textura, mm_por_px, K, tam, rvec, t_d)
    return izq, der, tomates
