# AgroVisión

**Sistema inteligente de detección, conteo, estimación de tamaño y clasificación de madurez de frutos mediante visión computacional estereoscópica.**

Proyecto universitario en Python + OpenCV. Detecta frutos en fotografías, los cuenta, los clasifica por etapa de madurez y, con visión estéreo calibrada, estima su **profundidad** y su **tamaño físico**. Lo que depende de la fruta está en un **perfil** (`perfiles/<fruta>.json`); hoy hay dos: **tomate** (Verde / Pintón / Maduro) y **banano** (Verde / Pintón / Maduro / Sobremaduro).

---

## 1. Instalación (Windows + VS Code)

```powershell
cd AgroVision
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name agrovision --display-name "Python (AgroVision)"
```

En VS Code: abra la carpeta `AgroVision`, abra `AgroVision.ipynb` y elija el kernel **Python (AgroVision)** (o el intérprete `.venv`). Instale las extensiones *Python* y *Jupyter* si VS Code lo pide.

Pruebas automáticas:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## 2. Estructura

```
AgroVision/
├── AgroVision.ipynb          Notebook principal (8 etapas, en español)
├── requirements.txt
├── README.md
├── perfiles/                 Un .json por fruta: colores, forma y etapas de madurez
│   ├── tomate.json
│   └── banano.json
├── herramientas/             banco.py (mide P/R/F1), buscar_parametros.py (ajusta un perfil),
│                             probar_camara.py y revisar_tablero.py (revisa fotos del tablero)
├── agrovision/               Código reutilizable
│   ├── config.py             Rutas (pathlib), parámetros y carga de perfiles
│   ├── procesamiento.py      Carga, espacios de color, filtros, CLAHE, textura
│   ├── segmentacion.py       inRange, Canny, morfología, contornos, watershed, validación
│   ├── caracteristicas.py    Descriptores geométricos y de color por tomate
│   ├── clasificacion.py      Reglas de madurez + SVM / k-NN / árbol
│   ├── estereo.py            Calibración, stereoCalibrate, rectificación, SGBM, 3D
│   ├── captura.py            Captura con dos cámaras (PC + celular)
│   ├── visualizacion.py      Gráficos y dibujos
│   └── evaluacion.py         Conteo, precisión/recall, matriz de confusión, errores, tiempos
├── tests/                    Pruebas con datos sintéticos de respuesta conocida
├── data/
│   ├── imagenes/             Fotos para el Modo 1
│   ├── estereo/izquierda/    Vistas izquierdas (Modo 2; Modo 3 con prefijo m3_)
│   ├── estereo/derecha/      Vistas derechas (mismo nombre/orden que la izquierda)
│   └── calibracion/
│       ├── celular/          Fotos del tablero con el celular (Modo 2)
│       └── par/izquierda|derecha/  Pares del tablero con PC + celular (Modo 3)
├── resultados/{detecciones,segmentaciones,disparidad,estadisticas}/
└── modelos/                  Calibraciones (.npz) y clasificador entrenado
```

## 3. Cómo usarlo

### Modo 1 — Fotografía individual
1. Copie la foto en `data/imagenes/`.
2. En la Etapa 1 elija la fruta (`FRUTA = "tomate"` o `"banano"`) y, si hay varias fotos, `NOMBRE_IMAGEN`.
3. Ejecute el Notebook desde el inicio (Etapas 1 a 4 y 8).
4. Si la detección no es buena, ajuste el perfil (ver abajo). Las figuras de la Etapa 3 muestran cada máscara y el motivo de cada rechazo para guiar el ajuste.

### Perfiles de fruta
Un perfil tiene dos secciones:

* `segmentacion`: las **clases de color** (`colores`: nombre → lista de intervalos HSV), cuáles de ellas se confunden con las hojas (`colores_follaje`, normalmente el verde: pasan por el filtro de textura y umbrales de forma más estrictos) y cualquier parámetro de `ParametrosSegmentacion` que cambie para esa fruta (forma, morfología, watershed).
* `madurez`: las clases de color con las que se cuentan los píxeles del fruto, las **etapas** (cada una con porcentajes `minimos`/`maximos` por clase; gana la primera que se cumple) y la etapa `por_defecto`.

**Agregar una fruta nueva:** copie el perfil de la fruta más parecida (redonda → `tomate.json`, alargada → `banano.json`), cambie colores y etapas, anote las cajas de 1 o 2 fotos en `data/anotaciones_cajas.csv` y ajuste con:

```powershell
.\.venv\Scripts\python herramientas/buscar_parametros.py 400 --fruta <fruta> --imagen <foto>
.\.venv\Scripts\python herramientas/banco.py resultados/estadisticas/mejor_<fruta>.json --fruta <fruta> --imagen <foto> --vista vista.png
```

Cada fruta tiene su propio archivo de etiquetas (`data/etiquetas_madurez_<fruta>.csv`) y su propio clasificador (`modelos/clasificador_madurez_<fruta>.joblib`), porque sus columnas de color son distintas.

**Fotos de banano:** frutos sobre un fondo liso y claro (mantel, papel), sin que se toquen, con luz difusa. Los racimos (bananos pegados a lo largo) todavía no se separan bien: el watershed busca centros de frutos redondos.

### Modo 2 — Estéreo con un celular (experimental)
1. Imprima `data/calibracion/tablero_9x6_25mm_carta.pdf` a **tamaño real / 100 %** (la regla impresa debe medir 100 mm; se regenera con `revisar_tablero.py --pdf <ruta>`), péguelo sobre algo rígido y **mida un cuadro con regla** → `PATRON.tam_cuadro_mm`.
2. 15-25 fotos del tablero con el celular → `data/calibracion/celular/` (misma resolución y orientación que usará en la escena, sin zoom). Revíselas antes de calibrar:

   ```powershell
   .\.venv\Scripts\python herramientas/revisar_tablero.py --cuadro <mm medidos> --vista resultados/cobertura.png
   ```

   Dice qué fotos sirven, el RMS, qué zonas del encuadre faltan y da un veredicto (APTA / USABLE / NO APTA).
3. Par de la escena: foto izquierda, deslizar el celular horizontalmente sobre una regla fija (6-10 cm), foto derecha. Enfoque y exposición bloqueados. Escena estática.
4. Opción A: deje el tablero visible en ambas fotos (`METODO_POSE_M2 = "tablero"`). Opción B: indique la distancia desplazada en `LINEA_BASE_M2_MM` (`"esencial"`).

> No equivale a dos cámaras simultáneas: cualquier cambio de la escena entre fotos produce errores.

### Modo 3 — Cámara del PC + celular
1. Monte ambas cámaras rígidamente, paralelas, con campos de visión superpuestos; mida la separación (`LINEA_BASE_MEDIDA_M3_MM`).
2. Conecte el celular como cámara. Recomendado: **DroidCam o Iriun por USB** (Android e iPhone; Windows lo ve como otra webcam, `ORIGEN_DER = 1` o `2`). Por Wi-Fi: *IP Webcam* (Android) → `ORIGEN_DER = "http://IP:8080/video"`, con más retraso (100-300 ms). Bloquee el enfoque del celular.
3. Antes de calibrar, compruebe la conexión y el desfase con `herramientas/probar_camara.py`:

   ```powershell
   .\.venv\Scripts\python herramientas/probar_camara.py --listar     # qué índices locales hay
   .\.venv\Scripts\python herramientas/probar_camara.py 1            # solo el celular
   .\.venv\Scripts\python herramientas/probar_camara.py 0 1          # PC + celular: desfase y veredicto
   ```

   Muestra la resolución, los FPS y el desfase, y al final dice si el par es apto (desfase ≤ 80 ms). Para medir el retraso real del celular, apúntelo a la ventana: la diferencia entre el reloj grande y el que se ve dentro de la imagen es el retraso.
4. Configure `ORIGEN_IZQ`, `ORIGEN_DER`, `INTERCAMBIAR` y `TAM_M3`; active `CAPTURAR_CALIBRACION_M3 = True` y capture ~20 pares del tablero (`c` captura, `s` intercambia, `q` sale). Luego `CAPTURAR_ESCENA_M3 = True` para la escena.
5. Ejecute la calibración (individual + `cv2.stereoCalibrate`) y la Etapa 7.

> Sin disparo sincronizado por hardware: se descartan capturas con desfase > 80 ms entre llegadas de fotogramas y se recomiendan escenas estáticas. Si se mueve una cámara, hay que recalibrar.

## 4. Método (resumen para la sustentación)

**Detección (no solo por color):** HSV + `cv2.inRange` (rojo en dos intervalos, naranja, verde) → la máscara verde se restringe a zonas lisas (`cv2.threshold` sobre la desviación estándar local) y se corta por bordes `cv2.Canny` → apertura/cierre (`cv2.morphologyEx`) → componentes con `cv2.findContours` (`RETR_CCOMP`) → watershed (`cv2.distanceTransform` + `cv2.watershed`) para frutos pegados → unión de la parte verde y roja de un pintón → validación con área (`cv2.contourArea`; además, un área mínima *relativa* a la mediana de los frutos de la misma foto, que no depende de la distancia), circularidad (con `cv2.arcLength`), solidez, relación de aspecto (`cv2.boundingRect`) y ajuste a elipse (`cv2.fitEllipse`), más estricta para candidatos solo verdes.

**Madurez:** porcentaje de píxeles de cada clase de color del perfil dentro de la máscara (excluyendo reflejos y sombras) + reglas por etapa configurables. Con etiquetas manuales, se comparan SVM, k-NN y árbol de decisión mediante validación cruzada estratificada.

**Estéreo:** `cv2.calibrateCamera` (K, distorsión) → `cv2.stereoCalibrate` / `cv2.solvePnP` / `cv2.findEssentialMat`+`cv2.recoverPose` (R, T) → `cv2.stereoRectify` + `cv2.initUndistortRectifyMap` + `cv2.remap` → `cv2.StereoSGBM` → `cv2.reprojectImageTo3D` (Z = f·B/d). Tamaño físico = tamaño_px · Z / f usando la mediana de Z dentro de cada fruto.

## 5. Estado del proyecto

| Componente | Implementado | Verificado | Cómo se verificó / qué falta |
|---|---|---|---|
| Carga, preprocesamiento, espacios de color | ✅ | ✅ sintético | Notebook ejecutado de principio a fin |
| Segmentación, watershed, conteo | ✅ | ✅ sintético · ✅ 1 foto real por fruta | Sintético: 7/7 tomates. Foto real de tomates: F1 = 0.77 (5/8, 0 falsos). **Ajustado con una sola foto: falta validar con fotos nuevas** |
| Características y reglas de madurez | ✅ | ✅ sintético · ⏳ foto real | 7/7 categorías correctas en la escena sintética |
| Perfil banano | ✅ | ✅ sintético · ✅ 1 foto real | Foto real (`bananos_separados.jpeg`): 4/4, 0 falsos (F1 = 1.00). Bananos pegados: no se separan. **Falta probar bananos verdes y amarillos sin manchas** |
| SVM / k-NN / árbol | ✅ | ⏳ | Requiere etiquetas reales en `data/etiquetas_madurez_<fruta>.csv` |
| Calibración intrínseca | ✅ | ✅ sintético | Recupera f con error < 2 % (RMS ≈ 0.1 px) |
| `stereoCalibrate` (cámaras distintas) | ✅ | ✅ sintético | Línea base recuperada 100,0 mm (real 100) con f distintas |
| Pose por tablero / matriz esencial | ✅ | ✅ sintético · ❌ par real | Sintético: línea base ±3 %, error epipolar < 1 px. Primer par real (iPhone 15) inservible: el celular giró 21° entre fotos. **Repetir el par** |
| Calibración aproximada sin tablero (focal del EXIF/ficha) | ✅ | ✅ sintético | Sin tablero: profundidad ≈ +4 %, diámetro ~±10 % |
| Rectificación + SGBM + profundidad | ✅ | ✅ sintético | Plano a distancia conocida: Z ±2 %; cadena completa: Z ±5 %, diámetro ±8 % |
| Captura PC + celular | ✅ | ✅ webcam · ⏳ celular | Webcam del PC probada con `probar_camara.py` (720p a 8 FPS, 480p a 30 FPS). Falta conectar el celular |
| Evaluación (conteo, P/R, confusión, errores) | ✅ | ⏳ | Requiere conteos, anotaciones y medidas reales |

"Sintético" = imágenes generadas por las pruebas en `tests/` con geometría y colores conocidos. Comprueban que el código calcula correctamente; **no son resultados del sistema sobre tomates reales**.

**Limitación conocida:** los parámetros morfológicos están en píxeles a `lado_max = 1280`. En las pruebas sintéticas, los tomates verdes de ~55 px de diámetro (frutos lejanos o pequeños en la imagen) no fueron detectados, mientras que los de ~70 px o más sí; los rojos se detectaron en todas las escalas probadas. Con fotos reales probablemente haya que ajustar `kernel_apertura_verde`, `textura_max_verde` y los umbrales de Canny.

## 6. Archivos de referencia para la evaluación (opcionales)

| Archivo | Columnas |
|---|---|
| `data/etiquetas_madurez_<fruta>.csv` | Lo genera el Notebook; llenar `categoria_real` con una etapa del perfil |
| `data/anotaciones_cajas.csv` | `imagen,x,y,w,h,categoria` (píxeles de la foto original) |
| `data/medidas_reales.csv` | `imagen,id,diametro_real_mm,profundidad_real_mm` |
