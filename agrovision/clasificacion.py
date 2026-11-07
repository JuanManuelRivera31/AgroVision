"""Clasificación de madurez: reglas de color configurables y clasificador ML.

El clasificador ML solo se entrena con etiquetas reales asignadas manualmente
(archivo ``data/etiquetas_madurez_<fruta>.csv``). Si no hay suficientes
ejemplos, las funciones lo indican y no producen métricas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .caracteristicas import columnas_ml
from .config import ReglasMadurez


# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------
def clasificar_por_reglas(fila: pd.Series | dict, r: ReglasMadurez) -> str:
    """Etapa de madurez a partir de los porcentajes de color del fruto."""
    for etapa in r.etapas:
        if etapa.nombre != r.por_defecto and etapa.cumple(fila):
            return etapa.nombre
    return r.por_defecto


def aplicar_reglas(df: pd.DataFrame, r: ReglasMadurez) -> pd.DataFrame:
    df = df.copy()
    df["categoria"] = [clasificar_por_reglas(f, r) for _, f in df.iterrows()] \
        if len(df) else pd.Series(dtype=str)
    return df


# ---------------------------------------------------------------------------
# Etiquetado manual
# ---------------------------------------------------------------------------
def exportar_para_etiquetar(df: pd.DataFrame, nombre_imagen: str, reglas: ReglasMadurez,
                            archivo: Path) -> Path:
    """Agrega las detecciones de una imagen al CSV de etiquetado.

    La columna ``categoria_real`` queda VACÍA: debe llenarse a mano con una de
    las etapas del perfil mirando la imagen con los números de cada fruto.
    ``categoria_regla`` es solo referencia. Si la imagen ya estaba en el
    archivo, no se duplica (se conservan las etiquetas existentes).
    """
    cols = ["imagen", "id", "categoria_regla", "categoria_real"] + columnas_ml(reglas)
    nuevo = df.assign(imagen=nombre_imagen,
                      categoria_regla=df.get("categoria", ""),
                      categoria_real="")[cols]
    if archivo.exists():
        previo = pd.read_csv(archivo, encoding="utf-8-sig")
        if nombre_imagen in set(previo["imagen"]):
            print(f"'{nombre_imagen}' ya está en {archivo.name}; no se modifica.")
            return archivo
        nuevo = pd.concat([previo, nuevo], ignore_index=True)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    nuevo.to_csv(archivo, index=False, encoding="utf-8-sig")
    return archivo


def cargar_etiquetas(archivo: Path, reglas: ReglasMadurez) -> pd.DataFrame:
    """Filas cuya ``categoria_real`` es una etapa del perfil."""
    if not archivo.exists():
        return pd.DataFrame()
    df = pd.read_csv(archivo, encoding="utf-8-sig")
    df["categoria_real"] = df["categoria_real"].astype(str).str.strip()
    return df[df["categoria_real"].isin(reglas.categorias)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Clasificadores ML
# ---------------------------------------------------------------------------
def _modelos() -> dict:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "k-NN (k=3)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=4, random_state=0),
    }


def comparar_clasificadores(df_etiq: pd.DataFrame, reglas: ReglasMadurez,
                            min_por_clase: int = 5):
    """Validación cruzada estratificada de SVM, k-NN y árbol vs. reglas.

    Devuelve (tabla_resultados, predicciones_cv) o (None, motivo) si los datos
    no alcanzan. Las predicciones son "fuera de muestra" (cross_val_predict),
    así que las métricas no están infladas por evaluar sobre datos de entrenamiento.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    if df_etiq.empty:
        return None, "No hay etiquetas manuales todavía."
    conteos = df_etiq["categoria_real"].value_counts()
    if len(conteos) < 2 or conteos.min() < min_por_clase:
        return None, (f"Datos insuficientes: se requieren >= {min_por_clase} "
                      f"ejemplos por clase y al menos 2 clases. Conteo actual: "
                      f"{conteos.to_dict()}")

    X = df_etiq[columnas_ml(reglas)].to_numpy(float)
    y = df_etiq["categoria_real"].to_numpy()
    k = int(min(5, conteos.min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    preds = {"Reglas de color": aplicar_reglas(df_etiq, reglas)["categoria"].to_numpy()}
    for nombre, modelo in _modelos().items():
        preds[nombre] = cross_val_predict(modelo, X, y, cv=cv)

    tabla = pd.DataFrame([{
        "metodo": n,
        "exactitud": accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "n_muestras": len(y),
        "validacion": "sin entrenamiento" if n == "Reglas de color" else f"{k}-fold estratificado",
    } for n, p in preds.items()])
    return tabla, preds


def entrenar_y_guardar(df_etiq: pd.DataFrame, nombre_modelo: str, reglas: ReglasMadurez,
                       destino: Path) -> Path:
    """Entrena el modelo elegido con todas las etiquetas y lo guarda (joblib)."""
    import joblib
    columnas = columnas_ml(reglas)
    modelo = _modelos()[nombre_modelo]
    modelo.fit(df_etiq[columnas].to_numpy(float), df_etiq["categoria_real"])
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "nombre": nombre_modelo,
                 "columnas": columnas, "n_entrenamiento": len(df_etiq)}, destino)
    return destino


def clasificar_con_modelo(df: pd.DataFrame, ruta: Path) -> pd.DataFrame | None:
    """Aplica un modelo guardado. Devuelve None si no existe."""
    import joblib
    if not ruta.exists() or df.empty:
        return None
    # joblib usa pickle: cargar solo modelos generados por este mismo proyecto.
    paquete = joblib.load(ruta)
    df = df.copy()
    df["categoria_ml"] = paquete["modelo"].predict(df[paquete["columnas"]].to_numpy(float))
    return df
