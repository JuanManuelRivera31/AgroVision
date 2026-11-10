"""Pruebas automáticas con datos sintéticos de respuesta conocida.

Ejecutar desde la raíz del proyecto:  python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agrovision import estereo as est  # noqa: E402
from agrovision.caracteristicas import extraer_caracteristicas  # noqa: E402
from agrovision.clasificacion import aplicar_reglas  # noqa: E402
from agrovision.config import ParametrosSGBM, PatronCalibracion, cargar_perfil  # noqa: E402
from agrovision.evaluacion import iou, precision_recall  # noqa: E402
from agrovision.procesamiento import cargar_imagen, guardar_imagen  # noqa: E402
from agrovision.segmentacion import segmentar  # noqa: E402
import sinteticos as sin  # noqa: E402

TOMATE = cargar_perfil("tomate")


def detectar_y_clasificar(img, perfil=TOMATE):
    res = segmentar(img, perfil.segmentacion)
    return res, aplicar_reglas(extraer_caracteristicas(res, perfil.madurez), perfil.madurez)


# ---------------------------------------------------------------------------
# Etapas 1-4: carga, segmentación, conteo, características y reglas
# ---------------------------------------------------------------------------
def test_carga_ruta_con_tildes(tmp_path):
    img = np.zeros((10, 20, 3), np.uint8)
    ruta = guardar_imagen(tmp_path / "tomate_pintón_ñ.png", img)
    assert cargar_imagen(ruta).shape == (10, 20, 3)


def test_cargar_perfil_cambia_solo_los_colores_indicados():
    base = cargar_perfil("tomate").segmentacion
    p = cargar_perfil("tomate", colores={"verde": [[[27, 50, 50], [85, 255, 255]]]},
                      area_min_frac=0.006).segmentacion
    assert p.colores["verde"] == [((27, 50, 50), (85, 255, 255))]
    assert p.colores["rojo"] == base.colores["rojo"]
    assert p.area_min_frac == 0.006
    with pytest.raises(ValueError):
        cargar_perfil("tomate", parametro_inexistente=1)


def test_tomate_tapado_por_tallos_se_detecta_entero():
    """Tres tallos "muerden" un tomate (como los sépalos de la foto real). Con la
    envolvente convexa se detecta UN fruto con su caja completa; sin ella el
    watershed lo parte en trozos que no corresponden a ningún fruto."""
    img = np.full((480, 640, 3), (150, 150, 150), np.uint8)
    cv2.circle(img, (320, 240), 90, (30, 30, 200), cv2.FILLED)
    for a, b in [((200, 150), (285, 210)), ((440, 150), (355, 210)), ((320, 360), (320, 290))]:
        cv2.line(img, a, b, (30, 90, 30), 26)
    caja_real = (230, 150, 180, 180)

    res = segmentar(img, TOMATE.segmentacion)
    assert res.conteo == 1, [r.motivo for r in res.rechazadas]
    assert iou(res.detecciones[0].bbox, caja_real) > 0.9

    sin_env = cargar_perfil("tomate", usar_envolvente=False, redondez_envolvente_unico=0)
    trozos = segmentar(img, sin_env.segmentacion).detecciones
    assert all(iou(d.bbox, caja_real) < 0.5 for d in trozos)


def test_deteccion_y_clasificacion_bananos_sinteticos():
    """Bananos curvos y alargados: el perfil debe aceptar su forma y sus etapas."""
    img, bananos = sin.escena_bananos()
    res, df = detectar_y_clasificar(img, cargar_perfil("banano"))
    assert res.conteo == len(bananos), [r.motivo for r in res.rechazadas]
    pr = precision_recall([d.bbox for d in res.detecciones], [b for b, _ in bananos], 0.5)
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0
    cat_pred = dict(zip(df["id"], df["categoria"]))
    for i_p, i_g, _ in pr["emparejamientos"]:
        assert cat_pred[res.detecciones[i_p].id] == bananos[i_g][1]


def test_deteccion_y_clasificacion_escena_sintetica():
    img, tomates = sin.escena_tomates()
    res, df = detectar_y_clasificar(img)
    assert res.conteo == len(tomates), [r.motivo for r in res.rechazadas]

    # Cada tomate verdadero debe estar emparejado con una detección.
    gt = [(cx - r, cy - r, 2 * r, 2 * r) for cx, cy, r, _ in tomates]
    pred = [d.bbox for d in res.detecciones]
    pr = precision_recall(pred, gt, 0.5)
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0

    cat_pred = dict(zip(df["id"], df["categoria"]))
    for i_p, i_g, _ in pr["emparejamientos"]:
        assert cat_pred[res.detecciones[i_p].id] == tomates[i_g][3]
    # Los dos tomates que se tocan se separaron con watershed.
    assert df["por_watershed"].sum() == 2


def test_area_minima_relativa_descarta_fragmentos_sin_depender_de_la_escala():
    """Un disco rojo pequeño pasa el área mínima absoluta (fracción de la imagen)
    pero es mucho menor que los frutos de su foto: con ``area_min_rel`` se descarta.
    La misma escena a la mitad de tamaño conserva todos los frutos."""
    img, tomates = sin.escena_tomates()
    img = img.copy()
    for cx, cy in [(80, 330), (900, 330), (450, 650)]:
        cv2.circle(img, (cx, cy), 20, (30, 30, 200), cv2.FILLED)
    gt = [(cx - r, cy - r, 2 * r, 2 * r) for cx, cy, r, _ in tomates]

    discos = [(cx - 20, cy - 20, 40, 40) for cx, cy in [(80, 330), (900, 330), (450, 650)]]
    sin_filtro = segmentar(img, cargar_perfil("tomate", area_min_rel=0.0).segmentacion)
    assert precision_recall([d.bbox for d in sin_filtro.detecciones], discos, 0.5)["TP"] == 3

    res = segmentar(img, cargar_perfil("tomate", area_min_rel=0.25).segmentacion)
    assert precision_recall([d.bbox for d in res.detecciones], discos, 0.5)["TP"] == 0
    assert precision_recall([d.bbox for d in res.detecciones], gt, 0.5)["recall"] == 1.0
    assert any(r.motivo == "pequeña frente a los demás" for r in res.rechazadas)

    # A la mitad de tamaño, el filtro no quita ningún tomate real (solo ruido).
    mitad = cv2.resize(sin.escena_tomates()[0], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gt_mitad = [tuple(v // 2 for v in b) for b in gt]
    antes, despues = (precision_recall(
        [d.bbox for d in segmentar(mitad, cargar_perfil("tomate", area_min_rel=rel)
                                   .segmentacion).detecciones], gt_mitad, 0.5)
        for rel in (0.0, 0.25))
    assert despues["TP"] == antes["TP"] and despues["FP"] <= antes["FP"]


def test_segmentar_con_frutos_cortados_por_el_marco():
    """Un fruto pegado al borde no debe romper la segmentación.

    La unión verde + cálida de un pintón y el cierre por componente trabajan en
    un lienzo con relleno: el contorno resultante tiene que quedar recortado a
    la imagen, o los bbox se salen y los recortes dejan de coincidir.
    """
    img, centros = sin.escena_tomates_en_bordes()
    res, df = detectar_y_clasificar(img)
    alto, ancho = res.imagen.shape[:2]
    for d in res.detecciones:
        x, y, w, h = d.bbox
        assert x >= 0 and y >= 0 and x + w <= ancho and y + h <= alto, d.bbox
        assert d.mascara_local.shape == (h, w)
    for r in res.rechazadas:
        x, y, w, h = r.bbox
        assert x >= 0 and y >= 0 and x + w <= ancho and y + h <= alto, r.bbox
    # Las características se calculan sobre esos contornos sin salirse.
    assert len(df) == res.conteo


def _video_sintetico(ruta, n=60, tam=(320, 240)):
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 30, tam)
    for k in range(n):
        f = np.full((tam[1], tam[0], 3), (k * 4) % 255, np.uint8)
        escritor.write(f)
    escritor.release()
    return ruta


def test_tablero_imprimible_tiene_tamano_fisico_exacto(tmp_path):
    """El PDF es carta horizontal y, rasterizado a 100 ppp, cada cuadro mide 25 mm."""
    patron = PatronCalibracion((9, 6), 25.0)
    fig = est.figura_tablero_imprimible(patron)
    pdf = tmp_path / "tablero.pdf"
    fig.savefig(pdf)
    assert b"/MediaBox [ 0 0 792 612 ]" in pdf.read_bytes()

    png = tmp_path / "tablero.png"
    fig.savefig(png, dpi=100)
    ok, esq = est.detectar_tablero(cargar_imagen(png), patron)
    assert ok
    paso_px = np.median(np.linalg.norm(np.diff(esq.reshape(6, 9, 2), axis=1), axis=2))
    assert paso_px == pytest.approx(25.0 / 25.4 * 100, rel=0.01)


def test_calibracion_aproximada_sin_tablero(tmp_path):
    """Sin tablero, K sale de la focal equivalente a 35 mm (EXIF o ficha del celular)."""
    from PIL import Image

    K, tam = sin.camara_virtual(f=1100.0)
    f35 = 1100.0 * 43.267 / np.hypot(*tam)          # la que daría el EXIF de esa cámara
    cal = est.calibracion_aproximada(tam, f35)
    assert cal.K[0, 0] == pytest.approx(1100.0, rel=1e-3)
    assert cal.K[1, 1] == cal.K[0, 0]
    assert (cal.K[0, 2], cal.K[1, 2]) == pytest.approx(((tam[0] - 1) / 2, (tam[1] - 1) / 2))
    # Foto vertical: misma cámara, misma focal en píxeles.
    assert est.calibracion_aproximada(tam[::-1], f35).K[0, 0] == pytest.approx(cal.K[0, 0])

    con_exif, sin_exif = tmp_path / "con.jpg", tmp_path / "sin.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26                  # FocalLengthIn35mmFilm
    Image.new("RGB", (64, 48)).save(con_exif, exif=exif)
    Image.new("RGB", (64, 48)).save(sin_exif)
    assert est.f35_desde_exif(con_exif) == 26
    assert est.f35_desde_exif(sin_exif) is None


def test_revisar_fotos_del_tablero(tmp_path):
    """Detecta el patrón, aparta fotos sin tablero o con otra orientación y da el veredicto."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "herramientas"))
    import revisar_tablero as rt

    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    for i, (rvec, tvec) in enumerate(sin.poses_tablero(14)):
        guardar_imagen(tmp_path / f"t{i:02d}.png", sin.render_tablero(K, tam, rvec, tvec, patron))
    guardar_imagen(tmp_path / "vacia.png", np.full((tam[1], tam[0], 3), 200, np.uint8))
    vertical = cv2.rotate(sin.render_tablero(K, tam, *sin.poses_tablero(1)[0], patron),
                          cv2.ROTATE_90_CLOCKWISE)
    guardar_imagen(tmp_path / "vertical.png", vertical)

    inf = rt.revisar(sorted(tmp_path.glob("*.png")), patron)
    estados = dict(zip(inf["fotos"]["archivo"], inf["fotos"]["estado"]))
    assert estados["vacia.png"] == "sin patrón"
    assert estados["vertical.png"] == "otro tamaño u orientación"
    assert sum(e == "ok" for e in estados.values()) == 14
    assert inf["calibracion"].rms < 0.5
    # El error por foto es del mismo orden que el RMS (antes salía ~150 px por un
    # error de forma de los arreglos: (54, 2) frente a (54, 1, 2)).
    assert max(inf["calibracion"].errores_vista) < 0.5
    assert inf["veredicto"] in ("APTA", "USABLE")


def test_captura_y_prueba_de_camaras_con_video(tmp_path):
    """Un archivo de video se abre como una cámara de red: sirve para probar sin hardware."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "herramientas"))
    import probar_camara as pc
    from agrovision.captura import FuenteCamara, ParEstereo

    v = str(_video_sintetico(tmp_path / "a.avi"))
    f = FuenteCamara(v, "video")
    try:
        img, _ = f.ultimo()
        assert img.shape == (240, 320, 3)
    finally:
        f.cerrar()
    assert f.n_fotogramas > 0

    par = ParEstereo(v, str(_video_sintetico(tmp_path / "b.avi")))
    try:
        fi, fd, desfase = par.capturar()
        assert fi.shape == fd.shape and desfase >= 0
    finally:
        par.cerrar()

    informe = pc.probar([v], segundos=0.5, con_ventana=False, resolucion=None, escala=0.5)
    assert informe["fuentes"][0]["resolucion"] == (320, 240)
    assert pc.interpretar_origen("1") == 1
    assert pc.interpretar_origen("http://192.168.1.5:8080/video").startswith("http")
    r = pc.resumen_desfases([10, 20, 30, 200])
    assert r["aptas_%"] == 75 and pc.veredicto_par(r).startswith("USABLE")
    assert pc.veredicto_par(pc.resumen_desfases([10, 20])).startswith("APTO")


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


# ---------------------------------------------------------------------------
# Etapas 5-7: estéreo
# ---------------------------------------------------------------------------
def test_armonizar_y_verificar():
    a = np.zeros((3000, 4000, 3), np.uint8)
    b = np.zeros((720, 1280, 3), np.uint8)
    assert not est.verificar_par(a, b)["valido_sin_cambios"]
    assert est.armonizar(a, (1280, 720)).shape == (720, 1280, 3)


def test_calibracion_camara_recupera_K():
    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    obj = est.puntos_objeto(patron)
    obj_pts, img_pts = [], []
    for rvec, tvec in sin.poses_tablero():
        img = sin.render_tablero(K, tam, rvec, tvec, patron)
        ok, esq = est.detectar_tablero(img, patron)
        assert ok
        obj_pts.append(obj)
        img_pts.append(esq)
    cal = est.calibrar_desde_puntos(obj_pts, img_pts, tam)
    assert cal.rms < 0.5
    assert cal.K[0, 0] == pytest.approx(K[0, 0], rel=0.02)
    assert cal.K[0, 2] == pytest.approx(K[0, 2], abs=10)


def test_calibracion_estereo_recupera_linea_base():
    """Proyecta el patrón en dos cámaras con R, T conocidos y los recupera."""
    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    dist = np.zeros(5)
    R_true = cv2.Rodrigues(np.array([0.0, np.radians(-3), 0.0]))[0]
    T_true = np.array([[-120.0], [0.0], [0.0]])
    obj = est.puntos_objeto(patron)
    rng = np.random.default_rng(0)
    obj_pts, pi, pd_ = [], [], []
    for rvec, tvec in sin.poses_tablero(12):
        Rb = cv2.Rodrigues(rvec)[0]
        p_i, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        R2 = R_true @ Rb
        t2 = R_true @ tvec.reshape(3, 1) + T_true
        p_d, _ = cv2.projectPoints(obj, cv2.Rodrigues(R2)[0], t2, K, dist)
        obj_pts.append(obj)
        pi.append((p_i + rng.normal(0, 0.2, p_i.shape)).astype(np.float32))
        pd_.append((p_d + rng.normal(0, 0.2, p_d.shape)).astype(np.float32))
    cal_i = est.CalibracionCamara(K, dist.reshape(1, 5), tam, 0.0)
    cal_d = est.CalibracionCamara(K, dist.reshape(1, 5), tam, 0.0)
    ce = est.calibrar_estereo_desde_puntos(obj_pts, pi, pd_, cal_i, cal_d)
    assert ce.linea_base_mm == pytest.approx(120.0, rel=0.01)
    assert ce.rms < 0.5


def test_pose_por_tablero():
    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    rvec_i, tvec_i = np.array([0.1, -0.1, 0.0]), np.array([-100.0, -60.0, 650.0])
    # Vista derecha = cámara desplazada 80 mm a la derecha: el patrón se ve
    # 80 mm más a la izquierda en su sistema.
    tvec_d = tvec_i + np.array([-80.0, 0, 0])
    img_i = sin.render_tablero(K, tam, rvec_i, tvec_i, patron)
    img_d = sin.render_tablero(K, tam, rvec_i, tvec_d, patron)
    cal = est.CalibracionCamara(K, np.zeros((1, 5)), tam, 0.0)
    ce = est.pose_por_tablero(img_i, img_d, cal, patron)
    assert ce.linea_base_mm == pytest.approx(80.0, rel=0.03)


def test_disparidad_y_profundidad_plano_conocido():
    """Plano frontal a profundidad conocida: Z = f*B/d."""
    K, tam = sin.camara_virtual(800, 600, 700.0)
    B, d_true = 100.0, 40
    rng = np.random.default_rng(3)
    textura = cv2.resize(rng.integers(0, 255, (150, 210), dtype=np.uint8), (840, 600),
                         interpolation=cv2.INTER_NEAREST)
    textura = cv2.GaussianBlur(textura, (3, 3), 0)
    izq = cv2.cvtColor(textura[:, 0:800], cv2.COLOR_GRAY2BGR)
    der = cv2.cvtColor(textura[:, d_true:d_true + 800], cv2.COLOR_GRAY2BGR)

    cal = est.CalibracionEstereo(K, np.zeros((1, 5)), K, np.zeros((1, 5)),
                                 np.eye(3), np.array([[-B], [0.0], [0.0]]), tam)
    rect = est.calcular_rectificacion(cal)
    disp = est.calcular_disparidad(izq, der, ParametrosSGBM(num_disparidades=64))
    zona = disp[100:500, 150:700]
    assert np.nanmedian(zona) == pytest.approx(d_true, abs=0.5)
    pts = est.nube_3d(disp, rect.Q)
    z = np.nanmedian(pts[100:500, 150:700, 2])
    assert z == pytest.approx(700.0 * B / d_true, rel=0.02)


def test_cadena_metrica_completa():
    """Pose por tablero -> rectificación -> SGBM -> detección -> diámetro en mm.

    Lámina plana a ~800 mm con tomates de diámetro conocido (2*r px * 0.5 mm/px).
    """
    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    izq, der, tomates = sin.par_estereo_lamina(patron, K, tam, B=60.0, Z=800.0)
    cal = est.CalibracionCamara(K, np.zeros((1, 5)), tam, 0.0)
    ce = est.pose_por_tablero(izq, der, cal, patron)
    assert ce.linea_base_mm == pytest.approx(60.0, rel=0.03)

    rect = est.calcular_rectificacion(ce)
    ri, rd = est.rectificar_par(izq, der, rect)
    assert est.error_epipolar_vertical(ri, rd)["mediana_px"] < 1.0
    disp = est.calcular_disparidad(ri, rd)
    pts = est.nube_3d(disp, rect.Q)
    res, df = detectar_y_clasificar(ri)
    med = df.merge(est.medir_tomates(df, res.detecciones, res.escala, pts, rect.f_px), on="id")
    assert len(med) == len(tomates)
    assert med["profundidad_mm"].median() == pytest.approx(800, rel=0.05)
    reales = sorted(2 * r * 0.5 for *_, r, _ in tomates)
    estimados = sorted(med["diametro_equiv_mm"])
    for e, r in zip(estimados, reales):
        assert e == pytest.approx(r, rel=0.08)


def test_pose_por_esencial_y_rectificacion_no_calibrada():
    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    izq, der, _ = sin.par_estereo_lamina(patron, K, tam, B=60.0, Z=800.0)
    cal = est.CalibracionCamara(K, np.zeros((1, 5)), tam, 0.0)

    ce, info = est.pose_por_esencial(izq, der, cal, linea_base_mm=60.0)
    direccion = ce.T.ravel() / np.linalg.norm(ce.T)
    assert direccion[0] == pytest.approx(-1.0, abs=0.05)   # derecha = desplazada en +x
    rect = est.calcular_rectificacion(ce)
    ri, rd = est.rectificar_par(izq, der, rect)
    assert est.error_epipolar_vertical(ri, rd)["mediana_px"] < 1.0

    ri, rd, info_nc = est.rectificacion_no_calibrada(izq, der)
    assert est.error_epipolar_vertical(ri, rd)["mediana_px"] < 1.0


def test_medir_por_altura_fruto_apoyado():
    """Esfera de 70 mm sobre la mesa, lente a 350 mm: su contorno está a Z = 315 mm."""
    f = 961.5
    d_px = 70.0 * f / 315.0
    fila = {"id": 1, "diametro_equiv_px_orig": d_px, "ancho_bbox_px_orig": d_px, "alto_bbox_px_orig": d_px}
    m = est.medir_por_altura(pd.DataFrame([fila]), 350.0, f).iloc[0]
    assert m["diametro_equiv_mm"] == pytest.approx(70.0)
    assert m["profundidad_mm"] == pytest.approx(315.0)
    assert m["ancho_mm"] == pytest.approx(70.0)
    filas = [{**fila, "id": i, "diametro_equiv_px_orig": d_px * k} for i, k in enumerate((0.7, 1.0, 1.0, 1.3))]
    assert est.medir_por_altura(pd.DataFrame(filas), 350.0, f)["tamano_relativo"].tolist() == \
        ["Pequeño", "Mediano", "Mediano", "Grande"]


def test_calibracion_por_fov():
    K, tam = sin.camara_virtual(f=1100.0)
    fov_h = np.degrees(2 * np.arctan(tam[0] / 2 / 1100.0))
    assert est.calibracion_por_fov(tam, fov_h).K[0, 0] == pytest.approx(1100.0, rel=1e-6)
    fov_d = np.degrees(2 * np.arctan(np.hypot(*tam) / 2 / 1100.0))
    assert est.calibracion_por_fov(tam, fov_d, "diagonal").K[0, 0] == pytest.approx(1100.0, rel=1e-6)
    # Foto 4:3 a pantalla completa: misma focal por f35 (diagonal) que por FOV horizontal.
    f35 = 1100.0 * 43.267 / np.hypot(*tam)
    fov_f35 = est.fov_horizontal_desde_f35(f35, aspecto_sensor=tam[0] / tam[1])
    assert fov_f35 == pytest.approx(fov_h, rel=1e-4)


def test_pose_por_esencial_dos_camaras_distintas_sin_tablero():
    """Modo 3: webcam del PC (izquierda) y celular por DroidCam (derecha) con focales distintas."""
    patron = PatronCalibracion((9, 6), 25.0)
    tam = (1280, 720)
    K_pc = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    K_cel = np.array([[1050.0, 0, 640], [0, 1050.0, 360], [0, 0, 1]])
    izq, der, _ = sin.par_estereo_lamina(patron, K_pc, tam, B=100.0, Z=800.0, K_der=K_cel)
    cal_pc = est.calibracion_por_fov(tam, np.degrees(2 * np.arctan(640 / 900.0)))
    cal_cel = est.calibracion_por_fov(tam, np.degrees(2 * np.arctan(640 / 1050.0)))

    ce, _ = est.pose_por_esencial(izq, der, cal_pc, 100.0, cal_der=cal_cel)
    assert ce.T.ravel()[0] / np.linalg.norm(ce.T) == pytest.approx(-1.0, abs=0.05)
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(ce.R)[0])) < 0.3
    rect = est.calcular_rectificacion(ce)
    ri, rd = est.rectificar_par(izq, der, rect)
    assert est.error_epipolar_vertical(ri, rd)["mediana_px"] < 1.0
    z = est.nube_3d(est.calcular_disparidad(ri, rd), rect.Q)[..., 2]
    assert np.nanmedian(z) == pytest.approx(800, rel=0.06)

    with pytest.raises(ValueError):
        est.pose_por_esencial(izq, der, cal_pc, 100.0, cal_der=est.calibracion_por_fov((640, 360), 60))


def test_pose_por_esencial_estable_y_profundidad_sin_tablero():
    """Sin tablero (K por focal equivalente) la rotación estimada debe ser estable:
    medio grado de error mueve la profundidad ~20 % con 6 cm de línea base."""
    patron = PatronCalibracion((9, 6), 25.0)
    K, tam = sin.camara_virtual()
    izq, der, _ = sin.par_estereo_lamina(patron, K, tam, B=60.0, Z=800.0)
    # Como llegan del celular: JPEG (el ruido de compresión es lo que desestabiliza).
    izq, der = (cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1],
                             cv2.IMREAD_COLOR) for im in (izq, der))
    cal = est.calibracion_aproximada(tam, round(1100.0 * 43.267 / np.hypot(*tam)))
    rotaciones = []
    for _ in range(6):
        ce, _ = est.pose_por_esencial(izq, der, cal, 60.0)
        rotaciones.append(np.degrees(np.linalg.norm(cv2.Rodrigues(ce.R)[0])))
    assert max(rotaciones) < 0.15
    rect = est.calcular_rectificacion(ce)
    ri, rd = est.rectificar_par(izq, der, rect)
    z = est.nube_3d(est.calcular_disparidad(ri, rd), rect.Q)[..., 2]
    assert np.nanmedian(z) == pytest.approx(800, rel=0.06)
