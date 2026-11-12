# AgroVisión — estado del proyecto (para el equipo)

*Actualizado: 11 de noviembre de 2026. Lo más reciente va
primero; debajo está la bitácora de cada sesión.*

## 11 de noviembre (noche): mangos reales y flujo de la foto

* **Mango con fotos reales de DroidCam:** el perfil exigía saturación ≥ 110 (ajustado con una foto de internet) y
  los mangos del celular tienen S ≈ 77–124, así que salían 2 cajas partidas en vez de 3 mangos. Se aflojaron los
  rangos de `perfiles/mango.json` (S ≥ 35–40, V ≥ 110–120; el fondo oscuro se descarta por brillo). Resultado en
  `foto_20261111_225957_mango.png`: 3/3, F1 1.00. En `mangos.jpg` F1 0.77 (antes ~0.80).
* **Reconocimiento:** esa foto salía "tomate". Se renombró a `_mango`, se agregó a `data/referencias_frutas.csv`
  y a `data/anotaciones_cajas.csv`; el reconocedor se reentrena solo. Ojo: ahora esa foto está en el entrenamiento,
  así que su acierto ya no es una validación; hacen falta más fotos de mango con otra luz.
* **Flujo de la cámara:** se quitó "siguiente foto". ESPACIO toma la foto, se ve congelada y se elige `g`/ENTER
  (guardar) o `r` (repetir); `q` cancela sin guardar. Solo al guardar sigue el notebook, que hace todo el análisis.
  `herramientas/tomar_foto.py` ahora solo captura y guarda (comparte `captura.tomar_foto`).

## 11 de noviembre: mango y manzana (pedido del profesor)

* **Perfiles nuevos:** `perfiles/mango.json` (Verde / Pintón / Maduro) y
  `perfiles/manzana.json` (Verde / Pintón / Madura), ajustados con `data/imagenes/mangos.jpg`
  y `manzanas.jpg` (fotos de internet).
* **Frutos pegados:** mangos y manzanas vienen apretados y la máscara de color los une en una
  sola mancha. Nueva opción del perfil `semillas_por_bordes`: se quitan los bordes de Canny
  (sobre la imagen muy suavizada) y cada trozo grande que queda es la semilla de un fruto;
  watershed las hace crecer dentro de la mancha. Tomate y banano no la usan (no cambian).
* **Medido** (IoU 0.5, cajas anotadas a mano, 4 versiones de la imagen con ruido/interpolación):
  mango F1 0.80–0.83 (7 de 8 mangos, 0 falsos); manzana F1 0.69–0.78 (canasta de ~18;
  un par de manzanas salen partidas en dos). Es UNA foto por fruta: es ajuste, no validación.
* **Imágenes pequeñas:** el notebook ahora amplía a 1280 px las fotos más chicas
  (`redimensionar(..., ampliar=True)`), igual que ya hacía `tomar_foto.py`.
* **La fruta se reconoce sola** (`agrovision/reconocimiento.py`): la foto se segmenta con
  el perfil de cada fruta, cada objeto se describe por color, forma y textura, y una SVM
  (clases balanceadas) entrenada con `data/referencias_frutas.csv` dice a qué fruta se
  parece; gana la fruta con más área. Validación dejando fuera cada foto (las de internet,
  por mitades): 10 de 12 aciertos. Reconoce bien las 9 fotos de referencia. Tarda ~1 s; el
  modelo se entrena solo la primera vez (`modelos/reconocedor_frutas.joblib`) y se
  reentrena si cambian las referencias o los perfiles.
  * **Para que reconozca mejor con el celular:** agregar al CSV una foto DroidCam de cada
    fruta (`foto_...png,mango`). Hoy mango, manzana y banano solo tienen fotos de internet.
* **Uso:** notebook → "Ejecutar todo", ESPACIO; para otra fruta, "Ejecutar todo" otra vez.
  Una sola fruta: `FRUTA = "mango"`. Sin notebook: `herramientas/tomar_foto.py`
  (tras el resultado `f` = siguiente foto, `q` = salir; `--fruta mango` la fija).

## Qué es

Proyecto universitario en Python + OpenCV **clásico** (sin redes neuronales). Se toma una
foto con el celular por DroidCam y el sistema detecta los frutos, los cuenta, clasifica su
madurez y estima su tamaño. Sirve para cualquier fruta: lo que depende de la fruta está en
`perfiles/<fruta>.json` (hoy hay `tomate`, `banano`, `mango` y `manzana`). Todo está en español. El README
explica el método completo.

## Estado para la entrega

**El flujo principal funciona de punta a punta**: foto por DroidCam → detección → conteo →
madurez → tamaño aproximado, todo evaluado en el notebook sobre la foto recién tomada.
**Listo para entregar.** Se probó con el celular real desde el notebook el 8 de noviembre y
funcionó. Quedan dos confirmaciones que mejoran el tamaño, pero no impiden entregar: ver
"Antes de entregar".

## Cómo correrlo

Primera vez, en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name agrovision --display-name "Python (AgroVision)"
.\.venv\Scripts\python -m pytest -q                       # deben pasar 26
```

Cada vez que se use:

1. Abrir la app DroidCam en el celular, con el celular y el PC en la misma red Wi-Fi.
   La IP que muestra la app debe coincidir con `ORIGEN_CELULAR` en `agrovision/config.py`
   (hoy `http://192.168.1.2:4747/video`).
2. Celular mirando hacia abajo, a unos 27 cm de la mesa.
3. Abrir `AgroVision.ipynb` en VS Code con el kernel **Python (AgroVision)** y pulsar
   **Run All**.
4. En la ventana de la cámara, pulsar **ESPACIO** para tomar la foto (`q` cancela y usa la
   última). El notebook sigue solo y evalúa esa foto.

Para evaluar una imagen ya guardada: en la Etapa 1, `TOMAR_FOTO = False` y su nombre en
`NOMBRE_IMAGEN`. Versión rápida sin notebook: `.\.venv\Scripts\python herramientas/tomar_foto.py`.

## Qué funciona (medido)

| Parte | Resultado | Cómo verlo |
|---|---|---|
| Foto real por DroidCam (5 tomates en bandeja) | 5/5 detectados, 0 falsos, IoU medio 0.98 contra cajas marcadas a ojo | Notebook con `foto_20261108_140128.png` |
| Madurez de esa foto | 3 Maduros, 2 Pintones. Sin etapa anotada: revisado solo a ojo | Notebook, Etapa 4 |
| Tamaño de esa foto | 5.0 a 5.8 cm. El tomate 1 es la referencia (5.5 cm), así que aún no hay validación independiente | Notebook, Etapa 7 |
| Segunda foto por DroidCam, tomates en otra posición, sin tocar parámetros | 5/5, error de conteo 0, cajas revisadas a ojo. El tomate amarillo da 5.5 cm en una foto y 5.8 cm en la otra | Notebook con `foto_20261108_145428.png` |
| Detección de tomates (`tomates.jpg`, foto real) | 7/8, 0 falsos, F1 = 0.93 | `herramientas/banco.py` |
| Madurez de tomates (`tomates.jpg`) | 7/7 contra la etapa anotada en las cajas | Notebook, Etapa 8.1 |
| Detección de bananos (foto real) | 4/4, 0 falsos, F1 = 1.00 | `herramientas/banco.py --fruta banano --imagen bananos_separados.jpeg` |
| Transmisión DroidCam por Wi-Fi | 1280x720, 30 fps | `herramientas/probar_camara.py` |
| Pruebas automáticas | 23/23 | `pytest` |
| Notebook | Ejecutado completo sin errores | — |

## Cómo se mide el tamaño

El celular queda a una altura fija H, mirando hacia abajo. H = 27.45 cm se dedujo de un
tomate de 5.5 cm medido con regla (`ALTURA_CELULAR_CM` en `agrovision/config.py`). Con el
modelo pinhole y el fruto apoyado en la mesa: **D = d_px · H / (f + d_px/2)**, donde f sale
de la focal del iPhone (26 mm equivalentes). Además, cada fruto sale como **Pequeño, Mediano
o Grande** (±10 % de la mediana de la foto), etiqueta que no depende de la altura. No se usa
tablero de calibración.

## Antes de entregar

- [ ] **Confirmar la referencia.** Se tomó como tomate de 5.5 cm el de arriba a la izquierda
      de `foto_20261108_140128.png` (id 1). Si fue otro, recalcular `ALTURA_CELULAR_CM`.
- [x] **Prueba con el celular** desde el notebook: foto tomada y evaluada completa (8 nov).
- [ ] **1 o 2 fotos nuevas** para comprobar la detección sin tocar parámetros (ya hay una).
- [ ] **Validar el tamaño:** medir con regla otro tomate distinto al de referencia y
      comparar con lo que da el sistema.
- [x] **Conteo manual:** `CONTEO_MANUAL = 5` en la Etapa 8.1. Si cambia la cantidad de frutos
      en la foto, hay que actualizarlo.

**Cuidado en la demo:** apuntar siempre a frutos. Una foto accidental del piso y un pie
(`foto_20261108_145132.png`) dio 12 detecciones falsas: tonos beige y piel caen en el rango
naranja, y el sistema no sabe que ahí no hay frutos.

## Opcional (no bloquea la entrega)

- [ ] Etiquetas para el clasificador ML: llenar `categoria_real` en
      `data/etiquetas_madurez_<fruta>.csv` (≥ 5 por clase) y correr la Etapa 4.1. Hoy el
      notebook muestra este componente como pendiente; la madurez se decide por reglas.
- [ ] Estéreo real (Modos 2 y 3): el código está y pasa las pruebas sintéticas, pero no
      tiene un par real válido. El notebook lo reporta como pendiente.
- [ ] Bananos verdes y amarillos sin manchas: solo se probaron bananos con manchas.
- [ ] Separar bananos pegados (`bananos.jpeg`): sin resolver.

## Limitaciones a declarar en la sustentación

- Los parámetros del perfil de tomate se ajustaron con dos fotos: es ajuste, no validación.
- El tamaño es **aproximado**: usa una sola referencia, la focal de ficha técnica y supone
  el celular a la misma altura en todas las fotos.
- El límite Pintón/Maduro depende del umbral elegido (Maduro = rojo ≥ 25 % y verde ≤ 15 %),
  porque la madurez se decide por color.
- En una escena sin frutos, con tonos beige o piel, aparecen detecciones falsas.
- Bananos que se tocan forman una sola mancha y no se separan.
- Un fruto mucho más lejano que el resto (menos de la mitad del área típica) se descarta
  por el filtro de área relativa.
- El clasificador ML y el estéreo real no están validados con datos reales.

## Herramientas (`herramientas/`)

| Script | Para qué |
|---|---|
| `tomar_foto.py` | Foto por DroidCam: ESPACIO toma, `g` guarda / `r` repite. El análisis va en el notebook |
| `probar_camara.py` | Prueba cámaras en vivo: resolución, FPS, desfase |
| `banco.py` | Mide precisión/recall/F1 de un perfil contra `data/anotaciones_cajas.csv` |
| `buscar_parametros.py` | Búsqueda aleatoria de parámetros de un perfil |
| `revisar_tablero.py` | Ya no se usa: el proyecto trabaja sin tablero |

---

# Prueba final con el celular — 8 nov 2026: FUNCIONÓ

Run All en el notebook → se abre la cámara de DroidCam → ESPACIO → el notebook evalúa esa
foto completa. Foto: `foto_20261108_145428.png` (1280x720, Wi-Fi), con los tomates en otra
posición que en la foto usada para ajustar y sin tocar parámetros.

| Tomate | Posición | Madurez | Tamaño relativo | Diámetro aprox. |
|---|---|---|---|---|
| 1 | Arriba izquierda | Maduro | Mediano | 5.5 cm |
| 2 | Arriba derecha | Maduro | Mediano | 5.3 cm |
| 3 | Centro (amarillo) | Pintón | Mediano | 5.8 cm |
| 4 | Abajo derecha | Maduro | Grande | 6.1 cm |
| 5 | Abajo izquierda | Pintón | Mediano | 5.2 cm |

- Conteo: 5 de 5, error 0 (`CONTEO_MANUAL = 5`).
- Repetibilidad: el tomate amarillo, el único reconocible a simple vista en las dos fotos,
  dio 5.5 cm y 5.8 cm: unos 3 mm de diferencia.
- Resultados guardados: `resultados/detecciones/foto_20261108_145428_detecciones.png` y
  `foto_20261108_145428_medidas.png`.
- Sin confirmar: qué tomate fue el de 5.5 cm (se asumió el de arriba a la izquierda de
  `foto_20261108_140128.png`) y la medida con regla de un segundo tomate.

---

# Foto directa con DroidCam — 8 nov 2026 (entrega: martes 10 nov)

`herramientas/tomar_foto.py`: vista en vivo, ESPACIO toma la foto y el sistema detecta,
cuenta y clasifica solo. Amplía las imágenes pequeñas a `lado_max` porque los parámetros
están en píxeles a esa escala. Simulando 640x480 con `tomates.jpg`: 5 de 6 detecciones
(4 sin ampliar) y madurez menos exacta que con la foto original.

Primera foto real por DroidCam (1280x720, 5 tomates en bandeja de plástico):
- Perdía el del centro (amarillo, tono 18-28) y los reflejos rojos del plástico inflaban
  las cajas. Perfil: rojo/naranja con S >= 130 y V >= 110, naranja hasta H = 28. Si varios
  frutos en corro forman una mancha redonda, ahora se dividen cuando cada parte es redonda.
- Resultado: 5/5, IoU medio 0.98. `tomates.jpg`: F1 0.93 (antes 0.86), 7/7 con la madurez
  anotada. Ajustado con estas dos fotos: falta validar con fotos nuevas.
- Madurez: Maduro = rojo >= 25 % y verde <= 15 %.
- El notebook toma la foto (`TOMAR_FOTO = True`) y evalúa la última; medida con una sola
  foto cenital: D = d_px·H / (f + d_px/2). H = 27.45 cm, deducida de un tomate de 5.5 cm
  medido con regla (el superior izquierdo). Tamaño aproximado: una sola referencia.

---

# Se elimina el tablero; Modo 3 con DroidCam — 5 nov 2026

Decisión: **no se imprime ni se usa tablero**. El estéreo se hace con la webcam del PC
(izquierda) + el celular por DroidCam (derecha). 22/22 pruebas; notebook sin errores.

- `estereo.calibracion_por_fov()`: K a partir del campo de visión (webcam: ficha técnica).
- `estereo.fov_horizontal_desde_f35()`: FOV horizontal del celular desde su focal
  equivalente (el video 16:9 conserva el ángulo horizontal del sensor 4:3).
- `estereo.pose_por_esencial(..., cal_der=)`: acepta una K distinta para la cámara derecha.
- Notebook: se quitó la celda que generaba el tablero; 6.1 explica la calibración sin
  tablero (tabla con/sin); 6.2 y 6.3 (Modo 2) solo usan focal + línea base; 6.4 (Modo 3)
  captura solo pares de la escena y calcula la pose del último par.
- Prueba nueva con dos cámaras de focales distintas (900 y 1050 px): Z ±6 %, error
  epipolar < 1 px.
- Punto para la sustentación: el tamaño físico = tamaño_px · B / d, así que un error en
  la focal casi no afecta los diámetros; sí afecta la profundidad Z.

---

# Primer par estéreo real (iPhone 15) — 1 nov 2026: NO SIRVE, hay que repetirlo

Fotos en `data/calibracion/par/izquierda/par1.jpeg` y `data/calibracion/par/derecha/par1.jpg.jpeg`
(quedaron en la carpeta del Modo 3, no en `data/estereo/`; se dejaron ahí como registro).

- Formatos distintos: izquierda 1600x900 (16:9), derecha 1600x1201 (4:3). Ambas sin EXIF
  (pasadas por WhatsApp).
- Pose estimada entre las fotos: **rotación de 21°** (debía ser ~0°) y corrimiento
  vertical de 125 px: el celular no se deslizó recto por el riel.
- Rectificación: error vertical **195 px** (debe ser < 1 px); ningún banano entero en la
  vista rectificada. No hay ajuste de código que lo rescate.
- Se agregó al notebook (Etapa 6.3) un AVISO cuando el giro estimado supera 5°.

# Cierre de la sesión del 1 nov 2026

- Notebook re-ejecutado completo con las salidas actuales (antes tenía salidas viejas):
  ahora analiza `bananos_separados.jpeg` con el perfil banano por defecto (antes tomaba
  `bananos.jpeg` con el perfil del tomate y daba 0 frutos).
- `requirements.txt`: se agregó `Pillow` (lectura del EXIF).
- README: tabla de estado actualizada con los resultados reales.

# Estéreo SIN tablero (no hay impresora) — 1 nov 2026

19/19 pruebas; notebook 24/24 sin errores con un par sintético sin fotos del tablero.

- **`estereo.calibracion_aproximada(tam, f35)`**: K a partir de la focal equivalente a
  35 mm (f_px = f35 · diagonal_px / 43.27, centro en el medio, sin distorsión).
  **`estereo.f35_desde_exif(ruta)`** la lee del EXIF (WhatsApp lo borra).
- Notebook, celda de la Etapa 6.2: si no hay fotos del tablero usa `F35_CELULAR` o el EXIF de
  la foto izquierda. `METODO_POSE_M2 = "esencial"` por defecto; falta poner `LINEA_BASE_M2_MM`.
- **Arreglo en `pose_por_esencial`:** la rotación estimada variaba de una ejecución a otra
  (hasta 0.5–0.8° con JPEG) porque FLANN es aleatorio y RANSAC solo es poco preciso;
  medio grado = ~20 % de error de profundidad. Ahora el emparejamiento es exacto
  (`BFMatcher`, determinista) y se usa RANSAC → LMEDS sobre los buenos.
  Par sintético (800 mm, B = 60 mm): rotación 0.04°, error vertical p90 de 9.2 a 0.18 px,
  Z ≈ 830 mm (+4 %).
- Precisión esperada sin tablero: profundidad ±5 %, diámetro ~±10 %. Declararlo en la
  sustentación y validar con un objeto de tamaño conocido en la escena.

**Configurado para iPhone 15:** `F35_CELULAR = 26`, `LINEA_BASE_M2_MM = 85.6` (largo de una
tarjeta bancaria, marcado con dos monedas) y `num_disparidades = 192`: con 85.6 mm y frutos a
45–70 cm el corrimiento es de 118 a 183 px. Deslizar el celular su largo (147.6 mm) daba
284 px a 50 cm, fuera del rango de SGBM.

**Pendiente (tú):** 2 fotos (izquierda/derecha) → `data/estereo/izquierda/` y `derecha/`
con el mismo nombre.

# Tablero de calibración — 1 nov 2026

17/17 pruebas. Todavía **no hay fotos del tablero** (falta imprimir y fotografiar).

- **`data/calibracion/tablero_9x6_25mm_carta.pdf`**: tablero a tamaño físico exacto
  (carta horizontal, cabe en A4), cuadros de 25 mm y regla de 100 mm para comprobar
  que la impresora no reescaló. El PNG anterior no tenía tamaño físico. Función:
  `estereo.figura_tablero_imprimible()`.
- **`herramientas/revisar_tablero.py`**: revisa las fotos ANTES de calibrar
  (patrón detectado, tamaño/orientación distintos, error por foto, cobertura 3x3 del
  encuadre, inclinación máxima, RMS) y da veredicto APTA / USABLE / NO APTA con consejos.
- **Bug corregido en `calibrar_desde_puntos`:** el error por foto (`errores_vista`)
  salía ~150 px en vez de ~0.07 px, porque `findChessboardCornersSB` devuelve las esquinas
  como (54, 2) y `projectPoints` como (54, 1, 2), y NumPy las combinaba en 54x54. El RMS
  global siempre estuvo bien; lo que estaba mal era la tabla por foto de la Etapa 6.2.
- Notebook (Etapa 6) y README apuntan al PDF y a la herramienta.

**Pasos (tú):** imprimir el PDF a tamaño real → comprobar la regla de 100 mm → pegarlo
en cartón → medir un cuadro → 15-25 fotos en `data/calibracion/celular/` →
`revisar_tablero.py --cuadro <mm> --vista resultados/cobertura.png`.

# Bananos separados (`data/imagenes/bananos_separados.jpeg`) — 1 nov 2026

**Perfil del banano ajustado con foto real: F1 = 1.00** (4/4, 0 falsos positivos);
escena sintética sigue 5/5; 15/15 pruebas.

- Cajas anotadas en `data/anotaciones_cajas.csv` (con pedúnculo), las 4 como Sobremaduro.
- Antes: 3/4 y 2 FP (una boronita amarilla sobre la bolsa y una mancha de la mesa
  oscura); el banano de la derecha se rechazaba por "poco circular" (contorno dentado
  por la raya oscura).
- Búsqueda de 400 combinaciones: 210 perfectas, y lo que decide es (a) eliminar lo
  pequeño y (b) un cierre morfológico ≥ 13. Se aplicó solo eso, con el área
  *relativa* para no depender de la distancia: `area_min_rel = 0.5`, `kernel_cierre = 15`.
  Estable en `area_min_rel` 0.3–0.5 y `kernel_cierre` 13–17.
- Madurez: café 35 %, 28 %, 20 % y 21 % → los 4 Sobremaduro.
- Sigue faltando: fotos con bananos **verdes y amarillos sin manchas** (hoy solo
  se probó una etapa) y una foto aparte para validar sin ajustar.
- `bananos.jpeg` (bananos pegados) sigue fallando: es la limitación de racimos.

# Primera foto real de bananos (`data/imagenes/bananos.jpeg`) — 1 nov 2026

- **Color: bien.** La máscara amarillo + café separa limpio los 4 bananos de la bolsa blanca.
  Café por banano: 13 %, 15 %, 34 % y 22 % (el umbral Maduro/Sobremaduro es 15 %).
- **Detección: falla.** Los 4 bananos se tocan, la limpieza morfológica rellena la
  rendija y queda una sola mancha (41 % de la imagen → "área excesiva"). Se probó
  sembrar el watershed con núcleos de brillo (V > 60, distancia > 0.3·máx): salen
  4 núcleos, pero las puntas que se tocan sin borde oscuro los unen y los cortes
  son irregulares. No se integró: sería ajustarlo a una sola foto.
- **Falso positivo:** la mesa de madera (esquina superior derecha) cae en el rango café.
- La foto llegó a 1200x1600 y 188 KB: parece reenviada por WhatsApp (el original del iPhone es 3024x4032).
- **Pendiente (tú):** repetir las fotos con los bananos separados 2-3 cm, solo la bolsa
  o mantel como fondo, en etapas distintas (verde, amarillo, con manchas) y pasadas por cable.

# Progreso — 1 de noviembre de 2026 (segunda parte)

**Hecho el punto 3: el perfil del tomate ya trae el ajuste (F1 = 0.77 sin pasar
ningún .json).** 15/15 pruebas, notebook 24/24 sin errores.

Copiar `perfil_ajustado.json` tal cual rompía 2 pruebas: con `area_min_frac = 0.006`
los tomates lejanos (prueba estéreo, unos 0.3 % de la imagen) se descartaban todos (0/7).
Ningún valor fijo servía a la vez para fotos de cerca y de lejos.

**Solución: área mínima relativa** (`area_min_rel`, nuevo en `ParametrosSegmentacion`).
Se descarta lo que mide menos de esa fracción del área *mediana* de los frutos
aceptados en la misma foto (motivo de rechazo: "pequeña frente a los demás"). No
depende de la distancia. En el tomate: `area_min_rel = 0.5`, `area_min_frac = 0.0015`
(antes 0.0008) y `ajuste_elipse_min_verde = 0.835`, más el resto de `perfil_ajustado.json`.

| Escena | Antes (por defecto) | `perfil_ajustado.json` | Perfil nuevo |
|---|---|---|---|
| Foto real (F1) | 0.40 | 0.77 | **0.77** (5 TP, 0 FP) |
| Escena sintética | 7/7 | 8/7 | **7/7** |
| Prueba estéreo (tomates lejanos) | 7/7 | 1/7 | **7/7** |

Limitaciones a declarar: si en una foto hay un fruto mucho más lejano que el resto
(menos de la mitad del área típica), se pierde. Y `ajuste_elipse_min_verde` solo
funciona bien entre 0.83 y 0.84: es un margen estrecho, fruto de ajustar sobre una
sola foto. El banano no usa el filtro (`area_min_rel = 0`). `herramientas/perfil_ajustado.json`
queda solo como registro.

---

# Progreso — sesión del 1 de noviembre de 2026

## Para retomar (lo primero que hay que leer)

Estado al cerrar: **todo funciona y está probado** (14/14 pruebas, notebook
24/24 celdas sin errores con tomate y con banano). Se hicieron los puntos 1
(perfiles de fruta) y 2 (script de prueba de cámara) del plan.

Pendiente, en orden:

1. **Fotos de bananos (tú):** 2-3 fotos, fondo liso y claro, bananos sin
   tocarse, en distintas etapas → `data/imagenes/`. Luego anotar sus cajas en
   `data/anotaciones_cajas.csv` y ajustar el perfil con
   `buscar_parametros.py 400 --fruta banano --imagen <foto>`.
2. **Conectar el celular (tú):** DroidCam/Iriun por USB →
   `probar_camara.py --listar` → `probar_camara.py 0 <índice>` → traer el veredicto.
   Decidir 640x480 o 1280x720 para la webcam (ver sección de cámaras abajo).
3. **Fijar en el perfil del tomate** los parámetros de `herramientas/perfil_ajustado.json`
   (F1 = 0.77): el notebook sigue usando los valores por defecto (F1 = 0.40).
4. **Tablero de calibración** para la parte estéreo (Vía B de la sesión del 31 oct, más abajo).

Para comprobar que todo sigue bien al volver:

```powershell
.\.venv\Scripts\python -m pytest -q                                      # 14 passed
.\.venv\Scripts\python herramientas/banco.py herramientas/perfil_ajustado.json   # F1=0.77
```

---

**El sistema ya no es solo para tomates.** Todo lo que dependía del tomate
(colores, categorías Verde/Pintón/Maduro, forma redonda) se movió a perfiles
`perfiles/<fruta>.json`. Hay dos: `tomate` y `banano`.

| Qué | Resultado |
|---|---|
| Tomate tras el refactor | **Idéntico** al de antes: mismas cajas, categorías, porcentajes y rechazos en la foto real (con valores por defecto y con `perfil_ajustado.json`) y en las dos escenas sintéticas. F1 sigue en 0.77 |
| Banano (escena sintética nueva) | 5/5 bananos detectados, 0 falsos positivos, 4/4 etapas correctas (Verde, Pintón, Maduro, Sobremaduro) |
| Pruebas | 14/14 (3 nuevas: escena de bananos, carga de perfil con cambios parciales y captura con videos sintéticos) |
| Notebook | Corre completo con `FRUTA = "tomate"` y con `FRUTA = "banano"` (24/24 celdas, 0 errores) |

**Hallazgo:** la unión de las dos mitades de un fruto pintón exigía que la
unión fuera *más circular* que la parte madura. En un banano, unir las mitades
lo alarga y baja la circularidad, así que el pintón salía como dos frutos. Ahora
es el parámetro `union_mas_circular` (sigue activo en el tomate y apagado en el banano).

**Cambios de uso:**
- Notebook, Etapa 1: `FRUTA = "tomate"` / `"banano"`.
- `banco.py` y `buscar_parametros.py` aceptan `--fruta` e `--imagen`; la vista va con `--vista`.
- `buscar_parametros.py` guarda la mejor combinación en `resultados/estadisticas/mejor_<fruta>.json`.
- `data/etiquetas_madurez.csv` pasó a llamarse `data/etiquetas_madurez_tomate.csv`.

**Siguiente paso con los bananos:**
1. 2 o 3 fotos con bananos en distintas etapas, sobre fondo liso y claro y sin
   que se toquen → `data/imagenes/`.
2. Marcar sus cajas en `data/anotaciones_cajas.csv` (`imagen,x,y,w,h,categoria`).
3. `.\.venv\Scripts\python herramientas/buscar_parametros.py 400 --fruta banano --imagen <foto>`.
4. Los valores del perfil del banano son **iniciales**, calculados para la escena
   sintética: hay que ajustarlos con las fotos reales. Lo que más puede fallar: el
   café de las manchas frente a una mesa de madera, y las sombras del amarillo
   (V < 110), que hoy cuentan como café.

**Limitación nueva que hay que declarar:** los racimos de bananos pegados a lo
largo no se separan bien, porque el watershed busca centros de frutos redondos.

## Prueba de cámaras (`herramientas/probar_camara.py`)

Abre 1 o 2 cámaras (índice o URL), las muestra en vivo y mide la resolución,
los FPS, si la imagen se congela y el desfase entre las dos cámaras. Al final
dice si el par es APTO, USABLE o NO APTO para el Modo 3 (límite: 80 ms). Tecla
`g`: guarda la imagen en `resultados/capturas/`. Probado con una prueba
automática (videos sintéticos como fuente) y con la webcam real.

**Medido en este PC:** la webcam integrada (índice 0) solo entrega formato YUY2
(no MJPG): **1280x720 a 8 FPS** o **640x480 a 30 FPS**. A 8 FPS cada imagen
puede tener hasta 125 ms de antigüedad, lo que por sí solo ya supera el límite
de 80 ms de desfase. Hay que decidir cuando el celular esté conectado:
- 640x480 en la webcam (desfase bajo, menos detalle), o
- 1280x720 aceptando que se rechacen más capturas (la escena es estática, así
  que basta con pulsar `c` varias veces).

**Para conectar el celular (tú):**
1. Instalar DroidCam o Iriun en el celular y su cliente en el PC; conectar por USB.
2. `.\.venv\Scripts\python herramientas/probar_camara.py --listar` → anotar el índice nuevo (1 o 2).
3. `.\.venv\Scripts\python herramientas/probar_camara.py 0 <índice>` → mirar el veredicto.
4. Con un par apto, poner esos índices en `ORIGEN_IZQ` / `ORIGEN_DER` (Etapa 6.4 del notebook).

---

# Progreso — sesión del 31 de octubre de 2026

Estado al cerrar: **el notebook corre completo sin errores** (24/24 celdas) y la
detección sobre la foto real pasó de **F1 = 0.12 a F1 = 0.77**. Falta terminar
el ajuste fino y, sobre todo, **tomar las fotos del celular** para la parte
estéreo, que es lo único que no se puede avanzar sin hardware.

---

## 1. Qué se arregló hoy

### 1.1 El error que impedía correr el notebook (resuelto)

**Síntoma:** `cv2.error (-209: Sizes of input arguments do not match)` en la
celda de la Etapa 3, y luego cinco `NameError` en cascada porque `res` y `E`
nunca llegaban a definirse.

**Causa raíz:** en `agrovision/segmentacion.py`, dos funciones construyen
contornos sobre un lienzo *con relleno* y los devuelven a coordenadas de imagen
sin recortarlos:

- `cerrar_componente()` — rellena con el tamaño del kernel.
- `_unir_pintones()` — rellena con `holgura` px.

Cuando un fruto queda cortado por el marco de la foto, el contorno resultante se
sale de la imagen. Con tu foto salía un bbox `(530, 894, 80, 73)` sobre una
imagen de 1280×960: `894 + 73 = 967 > 960`. NumPy recorta el *slice* en silencio
a `(66, 80)`, pero la máscara se creaba de `(73, 80)`, y `cv2.bitwise_and`
recibía dos matrices de distinto tamaño.

**Arreglo:** se añadió `_recortar_a_lienzo()` y se aplica en los dos sitios donde
nacen esos contornos (no en el `bitwise_and`, que era solo el síntoma). Esto
también protegía un fallo idéntico latente en `cv2.watershed`.

**Prueba de regresión:** `test_segmentar_con_frutos_cortados_por_el_marco`, con
el generador `escena_tomates_en_bordes()` que coloca pintones pegados a los
cuatro bordes. Falló primero con el error idéntico; pasa después.

### 1.2 El watershed no podía separar racimos (resuelto)

**Síntoma:** una sola caja gigante se tragaba 4 tomates que se tocan.

**Cómo se encontró:** un barrido de `umbral_dist_watershed` de 0.25 a 0.65 daba
**exactamente el mismo resultado**. Un parámetro que no cambia nada es la señal
de que el cuello de botella está en otra parte. Al instrumentar la función:

```
blob 587x417  area=115458  dmax=116  ventana_k=139px  semillas=2  -> regiones=2
```

Debían ser 4 semillas, no 2.

**Causa raíz:** la ventana de supresión de picos estaba **codificada a fuego**
como `k = 1.2 * dmax` = 139 px, más ancha que la distancia entre los centros de
frutos vecinos. Los picos vecinos se suprimían entre sí, quedaba una sola
semilla y el racimo no se dividía. Ningún parámetro del notebook la controlaba,
que es exactamente por qué el barrido salía plano.

**Arreglo:** se expuso como parámetro real `ventana_picos_frac` (en
`ParametrosSegmentacion`, por defecto **0.6**), documentando que un valor alto
fusiona racimos. Efecto medido: recall 0.12 → 0.50.

### 1.3 Diagnóstico de color: por qué fallaba con TU foto

Tu foto está **a contraluz**, y eso rompe la separación por tono. Medición del
tono HSV (mediana de un parche dentro de cada zona):

| Zona | H | S | V | Rango donde cae |
|---|---|---|---|---|
| Tomate verde izquierda | **28** | 216 | 88 | *naranja* (11–30) ← mal |
| Tomate verde centro | 33 | 164 | 105 | verde |
| **Hoja** arriba-derecha | **31** | 255 | 159 | verde |
| **Hoja** abajo | **30** | 197 | 224 | *naranja* ← mal |
| Tomate naranja real | 14 | 216 | 190 | naranja ✓ |
| Tomates rojos | 1–8 | 220–255 | 154–183 | rojo ✓ |

Las hojas y los tomates verdes tienen **el mismo tono** (H ≈ 28–36). La frontera
naranja/verde en H = 30 pasa justo por en medio de ambos grupos. Consecuencia:
hojas con H = 30 se clasificaban como "fruto cálido" y pasaban con los umbrales
de forma **permisivos** en vez de los estrictos, y aparecían como tomates.

Conclusión de diseño: **en esta foto el color no puede separar hoja de fruto
verde**. Lo que los separa es la textura (la hoja tiene nervaduras), el borde y
la forma — que es justo la maquinaria que el módulo ya tiene. El ajuste consiste
en mandar ambos al camino "verde" (subiendo `verde_bajo` y bajando
`naranja_alto`) y dejar que la textura y la forma decidan.

---

## 2. Métricas (foto real `data/imagenes/tomates.jpg`, 8 tomates, IoU 0.5)

| Configuración | Detec. | TP | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Como estaba al empezar | 8 | 1 | 7 | 7 | 0.12 | 0.12 | 0.12 |
| Con `ventana_picos_frac = 0.6` (valor actual) | 7 | 3 | 4 | 5 | 0.43 | 0.38 | 0.40 |
| **Mejor candidata encontrada** | 5 | 5 | **0** | 3 | **1.00** | 0.62 | **0.77** |

El "8 tomates" del inicio era una **coincidencia engañosa**: había 8 detecciones
y 8 tomates reales, pero solo 1 coincidía. El resto eran hojas, un tomate
partido en dos y una caja que fusionaba 4 frutos.

**Sin detectar todavía** con la mejor candidata: el rojo superior izquierda
(muy ocluido por el tallo), el rojo de la derecha y el verde pálido del centro
(casi tapado por otros frutos).

---

## 3. Archivos tocados

**Modificados**

| Archivo | Cambio |
|---|---|
| `agrovision/segmentacion.py` | `_recortar_a_lienzo()` nueva; `cerrar_componente()` y `_unir_pintones()` reciben la forma de la imagen y recortan; `separar_watershed()` acepta `ventana_frac` |
| `agrovision/config.py` | nuevo parámetro `ventana_picos_frac = 0.6` |
| `tests/sinteticos.py` | nuevo generador `escena_tomates_en_bordes()` |
| `tests/test_agrovision.py` | nueva prueba `test_segmentar_con_frutos_cortados_por_el_marco` |

**Nuevos**

| Archivo | Para qué |
|---|---|
| `herramientas/banco.py` | mide precisión/recall/F1 de una configuración contra las anotaciones |
| `herramientas/buscar_parametros.py` | búsqueda aleatoria de parámetros con doble puntaje (foto real + escena sintética) |
| `herramientas/perfil_ajustado.json` | la mejor combinación encontrada hoy (F1 = 0.77), lista para pasarle al banco |
| `data/anotaciones_cajas.csv` | las 8 cajas reales marcadas a mano sobre la foto |

**Generados por las ejecuciones** (se pueden borrar sin problema): todo lo de
`resultados/`, `data/etiquetas_madurez.csv` y
`data/calibracion/patron_tablero_9x6.png`.

**Pruebas:** 11 de 11 pasando (`.\.venv\Scripts\python -m pytest -q`).

---

## 4. Cómo reproducir lo de hoy

Todo desde la raíz del proyecto, con el entorno virtual:

```powershell
# 1. Las pruebas automáticas (incluida la regresión del error del marco)
.\.venv\Scripts\python -m pytest -q

# 2. La medición de la detección tal como está configurada ahora
.\.venv\Scripts\python herramientas/banco.py

# 3. La mejor configuración encontrada, guardando una vista comparativa
.\.venv\Scripts\python herramientas/banco.py herramientas/perfil_ajustado.json --vista resultados/detecciones/vista.png
```

Los tres comandos quedaron probados en PowerShell. El perfil va en un archivo
.json y no escrito en la línea de comandos, porque PowerShell destroza las
comillas del JSON en línea.

En la vista que genera: **verde** = tomate real encontrado, **rojo** = tomate
real que se perdió, **amarillo** = lo que el sistema detectó.

---

## 5. Paso a paso para mañana

### Vía A — terminar el ajuste de la detección (es trabajo de código)

1. **Correr la búsqueda larga.** Hoy solo alcanzaron 30 muestras; con 400 debería
   encontrar algo mejor. Tarda un rato, se puede cortar con Ctrl+C sin perder nada:

   ```powershell
   .\.venv\Scripts\python herramientas/buscar_parametros.py 400
   ```

   El ranking queda en `resultados/estadisticas/busqueda_parametros.csv`.

2. **Fijar los parámetros ganadores** como valores por defecto en
   `ParametrosSegmentacion` (`agrovision/config.py`), y volver a correr
   `pytest` para confirmar que la escena sintética sigue dando 7/7.

3. **Atacar los 3 tomates que faltan.** Los tres están muy ocluidos, así que
   probablemente no sea cuestión de umbrales sino de que la máscara del fruto
   queda partida por el tallo que lo cruza. Hay que mirar las máscaras de esa
   zona antes de tocar nada.

4. **Cuidado con un detalle:** la mejor candidata usa `area_min_frac = 0.006`,
   siete veces el valor original. Eso elimina los fragmentos pequeños (por eso
   llega a cero falsos positivos), pero también significa que **un tomate
   pequeño o lejano dejará de detectarse**. Hay que decidir si conviene, o si es
   mejor una precisión algo menor con mayor cobertura.

### Vía B — la parte del celular (esto solo lo puedes hacer tú)

Es lo único que da **milímetros** en vez de píxeles. Sin esto, el proyecto se
queda en conteo y clasificación, sin tamaño físico ni profundidad.

1. **Imprimir el tablero.** Ya está generado en
   `data/calibracion/patron_tablero_9x6.png`. Imprimir **al 100 %** (sin
   "ajustar a página") y pegarlo sobre algo rígido. Si se dobla, la calibración
   sale mal.

2. **Medir un cuadro con regla** y poner el número en la Etapa 6.1 del notebook:

   ```python
   PATRON = cfg.PatronCalibracion(esquinas=(9, 6), tam_cuadro_mm=24.5)  # tu medida
   ```

   Toda la escala en milímetros del proyecto sale de este número. Si lo mides
   mal un 5 %, todos los diámetros saldrán mal un 5 %.

3. **15 a 25 fotos del tablero** con el celular → `data/calibracion/celular/`.
   Misma resolución y orientación que usarás para la escena, **sin zoom**.
   Variar distancia e inclinación (hasta ±30°) y que el tablero aparezca también
   en las esquinas del encuadre, completo en cada foto.

4. **El par de la escena.** Apoyar el celular contra una regla sobre una
   superficie plana:
   - Foto desde la izquierda → `data/estereo/izquierda/`
   - Deslizarlo 6–10 cm a la derecha sobre la regla, sin girarlo ni cambiar la altura
   - Foto desde la derecha → `data/estereo/derecha/`

   Tres cosas que deciden si funciona:
   - **Bloquear enfoque y exposición** antes de la primera foto (mantener
     presionado el punto de enfoque hasta que aparezca AE/AF lock). Si el
     celular reenfoca entre fotos, la distancia focal cambia y la calibración
     ya no aplica.
   - **Dejar el tablero visible dentro de la escena en ambas fotos**, más o
     menos a la misma distancia que los tomates.
   - **Mismo nombre de archivo en las dos carpetas** (`par01.jpg` en ambas): el
     código las empareja por orden alfabético.
   - La escena no se puede mover entre las dos fotos (nada de viento).

5. **Correr el notebook completo.** Qué mirar:
   - Etapa 6.2: el **RMS debe ser < 1 px**, idealmente < 0.5. Si sale mayor,
     repetir las fotos del tablero.
   - Etapa 6.3: error vertical tras rectificar **< 1 px**.
   - Etapa 7: ya aparecen `profundidad_mm` y `diametro_equiv_mm`.
   - Si sale `AVISO: profundidades negativas`, están invertidas izquierda y
     derecha: intercambiar las fotos de carpeta.

6. **Validar con regla.** Medir un tomate real y compararlo. Para que el
   notebook calcule el error solo, crear `data/medidas_reales.csv` con
   `imagen,id,diametro_real_mm,profundidad_real_mm`.

### Vía C — lo que se puede adelantar sin nada más

- **Más fotos anotadas.** El ajuste de hoy está hecho sobre **una sola foto con
  8 tomates**: el riesgo de sobreajuste es alto y hay que decirlo así en la
  sustentación. Con 3 o 4 fotos más, anotadas en `data/anotaciones_cajas.csv`,
  el ajuste sería creíble. Es lo más valioso que se puede hacer sin hardware.
- **Entrenar el clasificador.** Abrir `data/etiquetas_madurez.csv` y llenar la
  columna `categoria_real` con Verde / Pintón / Maduro mirando la foto. Se
  necesitan ≥ 5 por clase, o sea varias fotos. Después, la Etapa 4.1 compara
  SVM, k-NN y árbol con validación cruzada.
- **Error de conteo.** Poner `CONTEO_MANUAL = 8` en la Etapa 8.1.

---

## 6. Limitaciones que hay que declarar en la sustentación

- El ajuste de parámetros está hecho sobre **una sola fotografía**. No es una
  validación, es un ajuste; para validar hacen falta fotos distintas de las
  usadas para ajustar.
- El Modo 2 (dos fotos con el mismo celular) **no equivale** a dos cámaras
  simultáneas. Cualquier movimiento de la escena entre fotos introduce error.
- Los parámetros morfológicos están en píxeles a `lado_max = 1280`: cambiar la
  resolución de trabajo invalida el ajuste.
- Los tomates muy ocluidos (más de la mitad tapada) no se detectan, y eso es una
  limitación del enfoque por contorno, no un parámetro mal puesto.

---

## 2026-11-11 (noche) — mango con manchas y análisis automático

- **Mango:** en fotos reales (fondo oscuro, 3 mangos con manchas) salían 7 trozos. Causa: las
  semillas por bordes de Canny leían manchas/textura como bordes entre frutos. Ahora
  `fusion_contacto` (perfil mango = 0.5) une los trozos que comparten una frontera larga
  (mismo fruto) y deja separados los que solo se tocan por una cintura (frutos distintos), y
  `_recrecer_nucleos` devuelve a cada fruto los trozos finos que antes se perdían. Colores del
  mango con S/V mínimos más bajos (zonas pálidas). Resultado: 3/3 en las 4 fotos propias.
- **Reconocimiento:** se agregaron 4 fotos a `data/referencias_frutas.csv` (3 de mango, 1 de
  tomate). Validación dejando una foto afuera: tomate y mango (fotos propias) aciertan; banano,
  manzana y el mango de internet tienen una sola foto, así que no se pueden validar: **más fotos
  propias de cada fruta = mejor reconocimiento**.
- **Flujo:** `tomar_foto.py` ahora, al guardar con ENTER, ejecuta AgroVision.ipynb solo
  (TOMAR_FOTO=False, esa foto) y deja las salidas escritas en el archivo (`--sin-notebook` lo
  evita). La celda 1.1 no abre la cámara si hay una foto de menos de 2 min.

## 2026-11-11 (más tarde) — tomates, bananos, medida y resumen en terminal

- **Tomates en charola (5 reales, salían 4):** 3 tomates pegados se fundían en 1 y el tomate
  amarillo-rosado quedaba a medias. Perfil tomate: `semillas_por_bordes` + `fusion_contacto` 0.5 y
  color `amarillo` en la segmentación. Ahora 5/5 y las demás fotos de tomate no cambian.
- **Bananos pegados:** salía 1 bloque (área excesiva). Perfil banano: semillas por bordes
  (suavizado 5, apertura 9) + `fusion_contacto` 0.3; la fusión no une dos alargados lado a lado
  (`_sigue_alargado`). bananos.jpeg: 4/4 (el 4.º pierde un poco su punta).
- **Manzana:** sin cambios (la fusión empeora el canasto apretado); en las fotos de tomate como
  sustituto no sobrecuenta. Falta una foto real de manzanas en la alfombra para validar.
- **Medida:** `largo_mm` y `grosor_mm` (rectángulo rotado) además del diámetro; y se corrigió la
  focal: los píxeles medidos son los de la imagen de trabajo, no los de la foto original (daba
  error en fotos que no eran de 1280 px de lado).
- **Reconocimiento:** cada fruta se puntúa solo con objetos hallados por SU perfil que se le
  parecen. El mango de internet (mangos.jpg) quedó 50/50 con manzana (foto apretada de internet).
- **Resumen en la terminal:** `agrovision/resumen.py`; `tomar_foto.py` lo imprime al guardar y lo
  guarda en `resultados/estadisticas/<foto>_resumen.txt`.
