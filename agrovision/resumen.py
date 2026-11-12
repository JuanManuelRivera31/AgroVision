"""Análisis completo de UNA foto (reconocer, contar, madurez, tamaño) y resumen en texto.

Usa exactamente las mismas funciones que AgroVision.ipynb, así que el resumen de la
terminal coincide con lo que muestra el notebook. Sirve para ver el resultado apenas
se toma la foto (``herramientas/tomar_foto.py``) sin abrir el notebook.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import caracteristicas as car
from . import clasificacion as clas
from . import config as cfg
from . import estereo as est
from . import procesamiento as proc
from . import reconocimiento as rec
from . import segmentacion as seg


@dataclass
class Analisis:
    ruta: Path
    fruta: str
    reconocida: bool                 # True si la fruta se dedujo sola (no se forzó)
    puntajes: pd.DataFrame | None    # parecido a cada fruta (solo si se reconoció sola)
    perfil: cfg.PerfilFruta
    resultado: seg.ResultadoSegmentacion
    tabla: pd.DataFrame              # una fila por fruto: categoría, tamaño, medidas
    segundos: float
    altura_cm: float | None


def analizar_foto(ruta: Path, fruta: str | None = None,
                  altura_cm: float | None = cfg.ALTURA_CELULAR_CM) -> Analisis:
    """Reconoce la fruta (si ``fruta`` es None), la segmenta, clasifica y mide."""
    t0 = time.perf_counter()
    ruta = Path(ruta)
    original = proc.cargar_imagen(ruta)
    puntajes = None
    if fruta is None:
        r = rec.reconocer_fruta(original)
        fruta, puntajes = r.fruta, r.puntajes
    perfil = cfg.cargar_perfil(fruta)
    img, _ = proc.redimensionar(original, perfil.segmentacion.lado_max, ampliar=True)
    res = seg.segmentar(img, perfil.segmentacion)
    tabla = clas.aplicar_reglas(car.extraer_caracteristicas(res, perfil.madurez), perfil.madurez)
    if altura_cm and len(tabla):
        # Los píxeles de la tabla son los de ``img``: la focal se calcula para ese tamaño.
        f_px = est.calibracion_por_fov(
            (max(img.shape[:2]), min(img.shape[:2])),
            est.fov_horizontal_desde_f35(cfg.F35_CELULAR)).K[0, 0]
        tabla = tabla.merge(est.medir_por_altura(tabla, altura_cm * 10, f_px), on="id")
    return Analisis(ruta, fruta, puntajes is not None, puntajes, perfil, res, tabla,
                    time.perf_counter() - t0, altura_cm)


def texto_resumen(a: Analisis) -> str:
    """Resumen legible: fruta, conteo, madurez y tamaño de cada fruto."""
    t, reglas = a.tabla, a.perfil.madurez
    barra = "=" * 64
    L = [barra, f"  AGROVISION - RESULTADO   ({a.ruta.name})", barra]
    origen = "reconocida sola" if a.reconocida else "indicada a mano"
    L.append(f"Fruta           : {a.fruta.upper()} ({origen})")
    if a.puntajes is not None:
        p = a.puntajes[a.puntajes["puntaje"] > 0]
        L.append("Parecido        : " + ", ".join(f"{f} {s:.0%}" for f, s in zip(p["fruta"], p["puntaje"])))
    L.append(f"Frutos contados : {len(t)}")
    if t.empty:
        L += ["", "No se detectó ningún fruto. Revise la luz, el fondo y que los frutos no se tapen.", barra]
        return "\n".join(L)

    conteo = t["categoria"].value_counts().reindex(reglas.categorias, fill_value=0)
    L.append("Madurez         : " + "   ".join(f"{c} {n}" for c, n in conteo.items()))
    medido = "diametro_equiv_mm" in t and t["diametro_equiv_mm"].notna().any()
    if medido:
        d = t["diametro_equiv_mm"] / 10
        L.append(f"Diámetro medio  : {d.mean():.1f} cm   (mínimo {d.min():.1f}, máximo {d.max():.1f})")
        L.append(f"Lente a         : {a.altura_cm:g} cm de la superficie (tamaños aproximados)")
    L.append("")
    cab = f"{'#':>2}  {'Madurez':<12}"
    if medido:
        cab += f"{'Largo x ancho (cm)':<20}{'Diám. (cm)':<12}"
    cab += "Tamaño"
    L += [cab, "-" * len(cab)]
    for _, f in t.iterrows():
        fila = f"{int(f['id']):>2}  {f['categoria']:<12}"
        if medido:
            fila += f"{f['largo_mm'] / 10:>6.1f} x {f['grosor_mm'] / 10:<9.1f}  {f['diametro_equiv_mm'] / 10:<10.1f}"
        fila += f"{f.get('tamano_relativo', '-')}"
        L.append(fila)
    L += ["", f"Tiempo de análisis: {a.segundos:.1f} s", barra]
    return "\n".join(L)


def guardar_resumen(a: Analisis) -> Path:
    """Guarda el resumen junto a los demás resultados."""
    cfg.DIR_RES_ESTADISTICAS.mkdir(parents=True, exist_ok=True)
    ruta = cfg.DIR_RES_ESTADISTICAS / f"{a.ruta.stem}_resumen.txt"
    ruta.write_text(texto_resumen(a), encoding="utf-8")
    return ruta
