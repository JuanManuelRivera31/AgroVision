# AgroVisión

**Sistema inteligente de detección, conteo, estimación de tamaño y clasificación de madurez de frutos mediante visión computacional estereoscópica.**

Proyecto universitario en Python + OpenCV. Detecta frutos en fotografías, los cuenta, los clasifica por etapa de madurez y, con visión estéreo calibrada, estima su **profundidad** y su **tamaño físico**. Lo que depende de la fruta está en un **perfil** (`perfiles/<fruta>.json`); hoy hay cuatro: **tomate** (Verde / Pintón / Maduro), **banano** (Verde / Pintón / Maduro / Sobremaduro), **mango** (Verde / Pintón / Maduro) y **manzana** (Verde / Pintón / Madura).

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
│   ├── mango.json
│   ├── manzana.json
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
2. En la Etapa 1 la fruta **se reconoce sola** (`agrovision/reconocimiento.py`, SVM entrenada con `data/referencias_frutas.csv`); para trabajar con una sola fruta se fija con `FRUTA = "mango"`.
3. Ejecute el Notebook desde el inicio (Etapas 1 a 4 y 8).
4. Si la detección no es buena, ajuste el perfil (ver abajo). Las figuras de la Etapa 3 muestran cada máscara y el motivo de cada rechazo para guiar el ajuste.

**Foto directa con el celular (DroidCam):** `.\.venv\Scripts\python herramientas/tomar_foto.py` abre la cámara; ESPACIO toma la foto y `g`/ENTER la guarda (`r` repite). Al guardar, el sistema reconoce la fruta, cuenta, clasifica la madurez y mide, **imprime el resumen en la terminal** y escribe el análisis completo en `AgroVision.ipynb` sin tener que ejecutarlo (con `--sin-notebook` solo guarda la foto). El tamaño es aproximado: cm con el celular a la altura `ALTURA_CELULAR_CM` (`agrovision/config.py`, deducida de un tomate de 5.5 cm) y Pequeño/Mediano/Grande frente a los demás de la foto. La IP está en `ORIGEN_CELULAR`. También sirve "Ejecutar todo" en el notebook (abre la cámara si no hay una foto de menos de 2 min).

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

**El proyecto no usa tablero de calibración.** La focal sale de la ficha técnica (focal equivalente a 35 mm del celular, campo de visión de la webcam), la rotación entre vistas de la matriz esencial con puntos SIFT de la propia escena y la escala de la línea base medida con regla. Ver la Etapa 6.1 del notebook.

### Modo 2 — Estéreo con un celular (experimental)
1. Indique la focal equivalente del celular en `F35_CELULAR` (o use una foto original con EXIF).
2. Par de la escena: foto izquierda, deslizar el celular horizontalmente sobre una regla fija (6-10 cm), foto derecha. Enfoque y exposición bloqueados. Escena estática.
3. Indique la distancia desplazada en `LINEA_BASE_M2_MM`.

> No equivale a dos cámaras simultáneas: cualquier cambio de la escena entre fotos produce errores.

### Modo 3 — Cámara del PC + celular por DroidCam
1. Monte ambas cámaras rígidamente, paralelas, con campos de visión superpuestos; mida con regla la separación entre lentes (`LINEA_BASE_MEDIDA_M3_MM`).
2. Conecte el celular con **DroidCam por USB** (Android e iPhone; con el cliente de Windows se ve como otra webcam, `ORIGEN_DER = 1` o `2`; sin cliente, `ORIGEN_DER = "http://IP:4747/video"`, con más retraso). Bloquee el enfoque y no use zoom.
3. Compruebe la conexión y el desfase con `herramientas/probar_camara.py`:

   ```powershell
   .\.venv\Scripts\python herramientas/probar_camara.py --listar     # qué índices locales hay
   .\.venv\Scripts\python herramientas/probar_camara.py 1            # solo el celular
   .\.venv\Scripts\python herramientas/probar_camara.py 0 1          # PC + celular: desfase y veredicto
   ```

   Muestra la resolución, los FPS y el desfase, y al final dice si el par es apto (desfase ≤ 80 ms). Para medir el retraso real del celular, apúntelo a la ventana: la diferencia entre el reloj grande y el que se ve dentro de la imagen es el retraso.
4. Configure `ORIGEN_IZQ`, `ORIGEN_DER`, `INTERCAMBIAR` y `TAM_M3`; active `CAPTURAR_ESCENA_M3 = True` y capture pares de la escena (`c` captura, `s` intercambia, `q` sale). La escena debe tener textura para que SIFT encuentre correspondencias.
5. Indique `FOV_PC_GRADOS` (ficha de la webcam; ~70° horizontal si no aparece) y ejecute la Etapa 6.4 y la Etapa 7.

> Sin disparo sincronizado por hardware: se descartan capturas con desfase > 80 ms entre llegadas de fotogramas y se recomiendan escenas estáticas. Si se mueve una cámara, vuelva a medir la línea base; la pose se recalcula de cada par.

## 4. Método (resumen para la sustentación)

**Detección (no solo por color):** HSV + `cv2.inRange` (rojo en dos intervalos, naranja, verde) → la máscara verde se restringe a zonas lisas (`cv2.threshold` sobre la desviación estándar local) y se corta por bordes `cv2.Canny` → apertura/cierre (`cv2.morphologyEx`) → componentes con `cv2.findContours` (`RETR_CCOMP`) → watershed (`cv2.distanceTransform` + `cv2.watershed`) para frutos pegados → **oclusión**: un fruto convexo tapado por tallos o por otro fruto se valida y se mide con su envolvente convexa (`cv2.convexHull`); si la envolvente es redonda (área / círculo mínimo `cv2.minEnclosingCircle` ≥ 0.68) es un solo fruto tapado y no se divide; el color para la madurez se toma solo de los píxeles visibles → unión de la parte verde y roja de un pintón → validación con área (`cv2.contourArea`; además, un área mínima *relativa* a la mediana de los frutos de la misma foto, que no depende de la distancia), circularidad (con `cv2.arcLength`), solidez, relación de aspecto (`cv2.boundingRect`) y ajuste a elipse (`cv2.fitEllipse`), más estricta para candidatos solo verdes.

**Madurez:** porcentaje de píxeles de cada clase de color del perfil dentro de la máscara (excluyendo reflejos y sombras) + reglas por etapa configurables. Con etiquetas manuales, se comparan SVM, k-NN y árbol de decisión mediante validación cruzada estratificada.

**Estéreo (sin tablero):** K por focal equivalente / campo de visión → `cv2.findEssentialMat`+`cv2.recoverPose` (R, dirección de T; escala = línea base medida) → `cv2.stereoRectify` + `cv2.initUndistortRectifyMap` + `cv2.remap` → `cv2.StereoSGBM` → `cv2.reprojectImageTo3D` (Z = f·B/d). Tamaño físico = tamaño_px · Z / f usando la mediana de Z dentro de cada fruto.

## 5. Estado del proyecto

| Componente | Implementado | Verificado | Cómo se verificó / qué falta |
|---|---|---|---|
| Carga, preprocesamiento, espacios de color | ✅ | ✅ sintético | Notebook ejecutado de principio a fin |
| Segmentación, watershed, conteo | ✅ | ✅ sintético · ✅ 1 foto real por fruta | Sintético: 7/7 tomates. Foto real de tomates: F1 = 0.86 (6/8, 0 falsos; antes 0.77). Los 2 perdidos están tapados más de la mitad. **Ajustado con una sola foto: falta validar con fotos nuevas** |
| Características y reglas de madurez | ✅ | ✅ sintético · ✅ foto real | Sintético 7/7. Foto real de tomates: 6/6 frutos detectados con la etapa anotada (Verde 2, Pintón 1, Maduro 3) |
| Perfil banano | ✅ | ✅ sintético · ✅ 1 foto real | Foto real (`bananos_separados.jpeg`): 4/4, 0 falsos (F1 = 1.00). Bananos pegados: no se separan. **Falta probar bananos verdes y amarillos sin manchas** |
| SVM / k-NN / árbol | ✅ | ⏳ | Requiere etiquetas reales en `data/etiquetas_madurez_<fruta>.csv` |
| Calibración intrínseca | ✅ | ✅ sintético | Recupera f con error < 2 % (RMS ≈ 0.1 px) |
| `stereoCalibrate` (cámaras distintas) | ✅ | ✅ sintético | Línea base recuperada 100,0 mm (real 100) con f distintas |
| Pose por tablero / matriz esencial | ✅ | ✅ sintético · ❌ par real | Sintético: línea base ±3 %, error epipolar < 1 px. Primer par real (iPhone 15) inservible: el celular giró 21° entre fotos. **Repetir el par** |
| Calibración aproximada sin tablero (focal del EXIF/ficha) | ✅ | ✅ sintético | Sin tablero: profundidad ≈ +4 %, diámetro ~±10 % |
| PC + celular sin tablero (FOV + matriz esencial, K distinta por cámara) | ✅ | ✅ sintético · ⏳ DroidCam | Sintético con f = 900 y 1050 px: Z ±6 %, error epipolar < 1 px. Falta capturar con DroidCam |
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
