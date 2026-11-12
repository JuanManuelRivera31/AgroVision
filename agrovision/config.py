"""Rutas del proyecto y parámetros configurables de AgroVisión.

Todas las rutas son relativas a la raíz del proyecto (pathlib), por lo que el
proyecto funciona igual en Windows, Linux o macOS sin modificar nada.

Lo que depende de la fruta (colores, etapas de madurez, forma esperada) vive
en un perfil ``perfiles/<fruta>.json`` y se carga con ``cargar_perfil``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent

DATA = RAIZ / "data"
DIR_IMAGENES = DATA / "imagenes"
DIR_ESTEREO = DATA / "estereo"
DIR_EST_IZQ = DIR_ESTEREO / "izquierda"
DIR_EST_DER = DIR_ESTEREO / "derecha"
DIR_CALIBRACION = DATA / "calibracion"
# Modo 2: fotos del patrón tomadas con el celular (calibración intrínseca).
DIR_CALIB_CELULAR = DIR_CALIBRACION / "celular"
# Modo 3: pares simultáneos del patrón (cámara PC + celular).
DIR_CALIB_PAR_IZQ = DIR_CALIBRACION / "par" / "izquierda"
DIR_CALIB_PAR_DER = DIR_CALIBRACION / "par" / "derecha"

RESULTADOS = RAIZ / "resultados"
DIR_RES_DETECCIONES = RESULTADOS / "detecciones"
DIR_RES_SEGMENTACIONES = RESULTADOS / "segmentaciones"
DIR_RES_DISPARIDAD = RESULTADOS / "disparidad"
DIR_RES_ESTADISTICAS = RESULTADOS / "estadisticas"

DIR_MODELOS = RAIZ / "modelos"
DIR_PERFILES = RAIZ / "perfiles"

# Archivos opcionales de anotación manual (ver README, sección Evaluación).
# Las etiquetas de madurez van por fruta: ver PerfilFruta.archivo_etiquetas.
ARCHIVO_ANOTACIONES = DATA / "anotaciones_cajas.csv"
ARCHIVO_MEDIDAS_REALES = DATA / "medidas_reales.csv"

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def crear_estructura() -> None:
    """Crea (si no existen) todas las carpetas del proyecto."""
    for d in (DIR_IMAGENES, DIR_EST_IZQ, DIR_EST_DER, DIR_CALIB_CELULAR,
              DIR_CALIB_PAR_IZQ, DIR_CALIB_PAR_DER, DIR_RES_DETECCIONES,
              DIR_RES_SEGMENTACIONES, DIR_RES_DISPARIDAD,
              DIR_RES_ESTADISTICAS, DIR_MODELOS):
        d.mkdir(parents=True, exist_ok=True)


def listar_imagenes(carpeta: Path) -> list[Path]:
    """Imágenes de una carpeta ordenadas por nombre."""
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.iterdir()
                  if p.suffix.lower() in EXTENSIONES_IMAGEN)


def _rangos_hsv(colores: dict) -> dict:
    """{nombre: [(bajo, alto), ...]} con tuplas, venga de Python o de JSON (listas)."""
    return {nombre: [(tuple(bajo), tuple(alto)) for bajo, alto in rangos]
            for nombre, rangos in colores.items()}


# Celular por DroidCam (IP y puerto que muestra la app) y su focal equivalente a 35 mm.
ORIGEN_CELULAR = "http://192.168.1.2:4747/video"
F35_CELULAR = 26     # iPhone 15, cámara principal (1x)
# Altura del lente sobre la mesa, deducida de un tomate de 5.5 cm medido con regla
# (foto_20261108_140128). Solo vale con el celular a esa altura y mirando hacia abajo.
ALTURA_CELULAR_CM = 27.45


# ---------------------------------------------------------------------------
# Parámetros de segmentación y detección
# ---------------------------------------------------------------------------
@dataclass
class ParametrosSegmentacion:
    # Lado mayor al que se reescala la imagen para procesar (px).
    lado_max: int = 1280

    # Clases de color del fruto: {nombre: [(hsv_bajo, hsv_alto), ...]} en HSV
    # de OpenCV (H: 0-179, S y V: 0-255). Varios intervalos por clase sirven
    # para el rojo, cuyo tono "da la vuelta". Las define el perfil de la fruta.
    colores: dict = field(default_factory=dict)
    # Clases que también aparecen en el follaje (normalmente el verde). Sus
    # candidatos pasan por el filtro de textura y los umbrales "_verde".
    colores_follaje: tuple = ()

    # En adelante, "verde" en un nombre de parámetro = candidato cuyo color es
    # de follaje (clases de ``colores_follaje``).
    # Filtro de textura para candidatos verdes: los frutos son lisos y las
    # hojas tienen nervaduras/bordes. Desviación estándar local máxima (0-255).
    textura_ventana: int = 9
    textura_max_verde: float = 18.0
    # Umbrales de histéresis de Canny para cortar la máscara verde por bordes.
    canny_bajo: int = 20
    canny_alto: int = 60

    # Morfología (tamaño de los elementos estructurantes, px a lado_max).
    kernel_apertura: int = 5
    kernel_cierre: int = 9
    # Apertura adicional más fuerte para la máscara verde (elimina hojas finas).
    kernel_apertura_verde: int = 11

    # Filtros geométricos (fracción del área de la imagen procesada).
    area_min_frac: float = 0.0008
    area_max_frac: float = 0.25
    # Área mínima RELATIVA: fracción del área mediana de los frutos aceptados en
    # la misma foto. No depende de la distancia (un fruto lejano y sus vecinos se
    # achican juntos), así que descarta fragmentos sin perder frutos pequeños.
    # 0 = desactivado. Un fruto mucho más lejano que el resto también se pierde.
    area_min_rel: float = 0.0
    circularidad_min: float = 0.45      # 4*pi*A/P^2 (1 = círculo perfecto)
    solidez_min: float = 0.80           # A / A_envolvente_convexa
    aspecto_max: float = 2.2            # lado mayor / lado menor del bbox
    # Para los candidatos verdes se exige más forma (son los que se confunden con hojas).
    circularidad_min_verde: float = 0.60
    solidez_min_verde: float = 0.88
    # IoU región vs. elipse ajustada (cv2.fitEllipse), solo candidatos verdes.
    ajuste_elipse_min_verde: float = 0.90

    # Watershed: solo se aplica a regiones poco sólidas (posible fusión de frutos).
    usar_watershed: bool = True
    solidez_para_watershed: float = 0.90
    umbral_dist_watershed: float = 0.55  # fracción del máximo de la transformada de distancia
    # Ventana de supresión de picos, como fracción del máximo de la distancia.
    # Dos frutos cuyos centros estén más cerca que esta ventana dan una sola
    # semilla y NO se separan, así que un valor alto fusiona racimos.
    ventana_picos_frac: float = 0.6

    # Fusión de trozos: tras partir un candidato con semillas de bordes, dos trozos
    # vecinos se vuelven a unir si su frontera común mide al menos esta fracción del
    # diámetro del menor (son partes de un mismo fruto con manchas). 0 = desactivado.
    fusion_contacto: float = 0.0

    # Oclusión (frutos convexos, p. ej. tomate). Un tallo o un fruto vecino le
    # "muerde" la silueta al fruto; su envolvente convexa (cv2.convexHull)
    # recupera la forma completa. Solo para candidatos de color distintivo.
    usar_envolvente: bool = False
    # Redondez de la envolvente = su área / área del círculo mínimo que la
    # encierra (cv2.minEnclosingCircle). Si una región poco sólida tiene una
    # envolvente así de redonda es UN fruto tapado (no varios pegados: dos
    # círculos que se tocan dan ~0.55) y no se divide con watershed. 0 = nunca.
    redondez_envolvente_unico: float = 0.0

    # La parte verde y la distintiva de un fruto a medio madurar solo se unen
    # si la unión es más circular que la parte distintiva sola. Desactivar en
    # frutos alargados (banano): unir sus dos mitades alarga la forma.
    union_mas_circular: bool = True

    # Frutos del mismo color apretados unos contra otros (mango, manzana): la
    # máscara de color los une en una sola mancha y la transformada de distancia
    # no basta para separarlos. Lo que sí los separa es la línea de sombra entre
    # ellos: se quitan los bordes de Canny (sobre la imagen muy suavizada, para
    # ignorar las pecas de la cáscara) y cada trozo grande que queda es la
    # SEMILLA de un fruto. Watershed hace crecer las semillas dentro de la mancha.
    semillas_por_bordes: bool = False
    bordes_suavizado: int = 15          # lado del gaussiano previo a Canny (px)
    bordes_canny: tuple = (15, 40)      # umbrales de histéresis
    bordes_dilatacion: int = 7          # grosor de la línea de corte (px)
    semillas_apertura: int = 11         # apertura que separa semillas unidas por un hilo
    semillas_area_min: float = 0.5      # área mínima de una semilla, fracción de area_min

    def __post_init__(self):
        self.colores = _rangos_hsv(self.colores)
        self.colores_follaje = tuple(self.colores_follaje)
        self.bordes_canny = tuple(self.bordes_canny)
        desconocidos = set(self.colores_follaje) - set(self.colores)
        if desconocidos:
            raise ValueError(f"colores_follaje {sorted(desconocidos)} no están en colores")

    def como_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parámetros de clasificación de madurez por reglas
# ---------------------------------------------------------------------------
@dataclass
class Etapa:
    """Una etapa de madurez y la condición sobre los porcentajes de color (0-1).

    ``minimos={"rojo": 0.6}`` exige pct_rojo >= 0.6; ``maximos`` exige <=.
    """
    nombre: str
    color_bgr: tuple = (255, 255, 0)
    minimos: dict = field(default_factory=dict)
    maximos: dict = field(default_factory=dict)

    def cumple(self, pct: dict) -> bool:
        return (all(pct[f"pct_{c}"] >= v for c, v in self.minimos.items())
                and all(pct[f"pct_{c}"] <= v for c, v in self.maximos.items()))


@dataclass
class ReglasMadurez:
    # Clases de color para contar los píxeles del fruto: {nombre: [(hsv_bajo,
    # hsv_alto), ...]}. Cada píxel va a la PRIMERA clase que lo contiene, así
    # que el orden importa cuando los intervalos se solapan.
    colores: dict = field(default_factory=dict)
    # Píxeles con poca saturación o brillo (reflejos, sombras) no se cuentan.
    s_min_valido: int = 40
    v_min_valido: int = 40
    # Etapas en orden de maduración. Gana la primera cuya condición se cumple;
    # si ninguna se cumple, la etapa ``por_defecto``.
    etapas: list = field(default_factory=list)
    por_defecto: str = ""
    # Clases de color de los ejes x, y del gráfico "espacio de decisión".
    ejes_grafico: tuple = ()

    def __post_init__(self):
        self.colores = _rangos_hsv(self.colores)
        self.etapas = [e if isinstance(e, Etapa) else Etapa(**e) for e in self.etapas]
        for e in self.etapas:
            e.color_bgr = tuple(e.color_bgr)
        self.ejes_grafico = tuple(self.ejes_grafico)
        usados = {c for e in self.etapas for c in (*e.minimos, *e.maximos)} | set(self.ejes_grafico)
        if usados - set(self.colores):
            raise ValueError(f"Las reglas usan colores no definidos: {sorted(usados - set(self.colores))}")
        if self.por_defecto not in self.categorias:
            raise ValueError(f"por_defecto '{self.por_defecto}' no es una etapa")

    @property
    def categorias(self) -> list[str]:
        return [e.nombre for e in self.etapas]

    @property
    def columnas_pct(self) -> list[str]:
        return [f"pct_{c}" for c in self.colores]

    @property
    def colores_bgr(self) -> dict:
        return {e.nombre: e.color_bgr for e in self.etapas}

    @property
    def colores_mpl(self) -> dict:
        """Los colores de cada etapa en RGB normalizado para Matplotlib."""
        return {n: (r / 255, g / 255, b / 255) for n, (b, g, r) in self.colores_bgr.items()}

    def como_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Perfiles de fruta
# ---------------------------------------------------------------------------
@dataclass
class PerfilFruta:
    nombre: str
    descripcion: str
    segmentacion: ParametrosSegmentacion
    madurez: ReglasMadurez

    @property
    def archivo_etiquetas(self) -> Path:
        return DATA / f"etiquetas_madurez_{self.nombre}.csv"

    @property
    def archivo_modelo(self) -> Path:
        return DIR_MODELOS / f"clasificador_madurez_{self.nombre}.joblib"


def listar_perfiles() -> list[str]:
    return sorted(p.stem for p in DIR_PERFILES.glob("*.json"))


def cargar_perfil(nombre: str, **cambios) -> PerfilFruta:
    """Lee ``perfiles/<nombre>.json``.

    ``cambios`` reemplaza parámetros de segmentación (útil para ajustar). En
    ``colores`` se reemplazan solo las clases indicadas y se conservan las demás.
    """
    ruta = DIR_PERFILES / f"{nombre}.json"
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el perfil '{nombre}'. Disponibles: {listar_perfiles()}")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    seg = dict(datos.get("segmentacion", {}))
    validos = {f.name for f in fields(ParametrosSegmentacion)}
    if set(cambios) - validos:
        raise ValueError(f"Parámetros desconocidos: {sorted(set(cambios) - validos)}")
    if "colores" in cambios:
        cambios = {**cambios, "colores": {**seg.get("colores", {}), **cambios["colores"]}}
    seg.update(cambios)
    return PerfilFruta(nombre=datos.get("nombre", nombre),
                       descripcion=datos.get("descripcion", ""),
                       segmentacion=ParametrosSegmentacion(**seg),
                       madurez=ReglasMadurez(**datos["madurez"]))


@dataclass
class ParametrosSGBM:
    """Parámetros de cv2.StereoSGBM (ver documentación de OpenCV)."""
    min_disparidad: int = 0
    num_disparidades: int = 128   # múltiplo de 16
    tam_bloque: int = 5           # impar, 3..11
    uniqueness: int = 10
    speckle_ventana: int = 100
    speckle_rango: int = 2
    disp12_max_diff: int = 1
    modo: str = "SGBM_3WAY"       # "SGBM", "HH", "SGBM_3WAY"

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatronCalibracion:
    """Tablero de ajedrez. `esquinas` = esquinas INTERNAS (columnas, filas)."""
    esquinas: tuple = (9, 6)
    # Medir con regla el lado de un cuadro DESPUÉS de imprimir (milímetros).
    tam_cuadro_mm: float = 25.0
