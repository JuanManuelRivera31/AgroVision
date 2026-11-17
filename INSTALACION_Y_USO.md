# AgroVisión — Guía de instalación y de prueba

Sistema en **Python + OpenCV clásico** que, a partir de una foto, **reconoce la fruta sola**, **cuenta** los frutos, clasifica su **madurez** y estima su **tamaño**.

**Frutas con las que trabaja:** 🍅 **tomate** · 🍌 **banano** · 🥭 **mango** · 🍎 **manzana**
(cada una tiene su perfil en `perfiles/<fruta>.json`).

**Contenido de esta guía**

1. [Opción rápida: ver resultados sin instalar nada](#1-opción-rápida-ver-resultados-sin-instalar-nada)
2. [Requisitos del computador](#2-requisitos-del-computador)
3. [Instalación paso a paso](#3-instalación-paso-a-paso) (Python → proyecto → `INSTALAR.bat` → verificar)
4. [Probar el sistema SIN celular](#4-probar-el-sistema-sin-celular)
5. [Probar CON el celular (DroidCam)](#5-probar-con-el-celular-droidcam)
6. [Usar el notebook en VS Code (opcional)](#6-usar-el-notebook-en-vs-code-opcional)
7. [Problemas frecuentes](#7-problemas-frecuentes)
8. [Qué hay en la carpeta](#8-qué-hay-en-la-carpeta)

---

## 1. Opción rápida: ver resultados sin instalar nada

Abra el archivo **`AgroVision_resultados.html`** con cualquier navegador (Chrome, Edge…). Es el notebook ya ejecutado, con todas las gráficas y el análisis. No necesita Python ni internet.

Si además quiere **ejecutar** el sistema, siga desde la sección 2.

---

## 2. Requisitos del computador

| Requisito | Detalle |
|-----------|---------|
| Sistema operativo | **Windows 10 u 11** de 64 bits |
| Espacio libre | ~3 GB (el proyecto ~90 MB + las librerías ~1.5 GB) |
| Memoria RAM | 4 GB o más (recomendado 8 GB) |
| Internet | Solo **durante la instalación** (descarga ~500 MB). Después funciona sin internet. |
| Permisos | No hace falta ser administrador |
| Para tomar fotos en vivo (opcional) | Un celular con la app **DroidCam** y una red Wi-Fi compartida con el PC |

**Qué se instala en total**

| # | Qué | Para qué | ¿Obligatorio? |
|---|-----|----------|---------------|
| 1 | **Python 3.12** | Ejecuta todo el sistema | **Sí** |
| 2 | Librerías de `requirements.txt` (OpenCV, NumPy, scikit-learn, pandas, matplotlib, Jupyter, pytest…) | El análisis de imágenes, el notebook y las pruebas | **Sí** — las instala `INSTALAR.bat` solo |
| 3 | App **DroidCam** (en el celular) | Usar el celular como cámara | Solo para fotos en vivo |
| 4 | VS Code + extensiones *Python* y *Jupyter* | Abrir el notebook en VS Code | No (hay alternativa en el navegador) |

---

## 3. Instalación paso a paso

### Paso 1 — Instalar Python 3.12

1. Entre a <https://www.python.org/downloads/> y descargue **Python 3.12** (botón “Download Python 3.12.x”; sirve cualquier 3.12; también funcionan 3.10, 3.11 y 3.13).
2. Ejecute el instalador. En la **primera pantalla**:
   * ✅ Marque la casilla de abajo **“Add python.exe to PATH”** (¡importante!).
   * Pulse **“Install Now”**.
3. Al final pulse, si aparece, **“Disable path length limit”** (evita errores con rutas largas) y luego **Close**.
4. **Verifique:** abra el menú Inicio, escriba `cmd`, abra “Símbolo del sistema” y escriba:

   ```
   python --version
   ```

   Debe responder algo como `Python 3.12.10`.

   > Si dice *“Python was not found; run without arguments to install from the Microsoft Store”*: abra *Configuración → Aplicaciones → Configuración avanzada de aplicaciones → Alias de ejecución de aplicaciones* y **desactive** los alias “python.exe” y “python3.exe”. Luego cierre y abra de nuevo el símbolo del sistema. (`INSTALAR.bat` usa además el lanzador `py`, que no tiene este problema.)

### Paso 2 — Descomprimir el proyecto

1. Copie `AgroVision_entrega.zip` al **Escritorio** (o a `C:\`).
2. Clic derecho → **Extraer todo…** → **Extraer**. (No lo ejecute desde dentro del zip: tiene que estar extraído.)
3. Debe quedar una carpeta `AgroVision` que contiene `INSTALAR.bat`, `EJECUTAR.bat`, `celular.txt`, etc.

> ⚠️ Use una **ruta corta** (`C:\Users\...\Desktop\AgroVision` o `C:\AgroVision`). Con carpetas dentro de carpetas muy largas, Windows puede fallar al instalar las librerías.
>
> Si descargó el zip de internet/Drive y Windows bloquea los archivos: clic derecho en el zip → *Propiedades* → marque **Desbloquear** → *Aceptar*, y extraiga de nuevo.

### Paso 3 — Ejecutar `INSTALAR.bat` (una sola vez)

1. Entre a la carpeta `AgroVision` y haga **doble clic en `INSTALAR.bat`**.
2. Si Windows muestra *“Windows protegió su PC”* (SmartScreen): pulse **“Más información” → “Ejecutar de todas formas”**. El archivo es solo un script de texto; puede abrirlo con el Bloc de notas para comprobarlo.
3. Se abre una ventana negra y hace esto, **sin que usted escriba nada**:

   | Lo que ve | Qué está haciendo | Tarda |
   |-----------|-------------------|-------|
   | `Python encontrado: Python 3.12.x` | Busca Python (prefiere 3.12) | segundos |
   | `Creando entorno virtual .venv ...` | Crea una carpeta `.venv` con un Python aislado solo para este proyecto | ~30 s |
   | `Instalando librerias ...` (muchas líneas “Downloading / Using cached”) | Descarga e instala las librerías de `requirements.txt` | 3–8 min |
   | `Registrando el kernel de Jupyter ...` | Deja listo el notebook | segundos |
   | `Verificando con las pruebas automaticas ...` | Corre las 28 pruebas | ~1 min |
   | **`Todas las pruebas pasaron.`** | Todo funciona | — |
   | `Instalacion terminada.` | Fin. Pulse una tecla para cerrar la ventana | — |

   **No cierre la ventana** hasta ver “Instalacion terminada”.

### Paso 4 — Comprobar que quedó bien

Debe haber cumplido las tres cosas:

* ✅ La ventana dijo **“Todas las pruebas pasaron.”**
* ✅ Existe la carpeta **`.venv`** dentro de `AgroVision`.
* ✅ Al hacer doble clic en `EJECUTAR.bat` aparece el menú con 5 opciones (sección 4).

Si algo falló, vaya a la [sección 7](#7-problemas-frecuentes).

<details>
<summary><b>Instalación manual</b> (equivalente, por si prefiere la terminal)</summary>

Abra PowerShell **dentro de la carpeta del proyecto** (en el Explorador, entre a `AgroVision`, haga clic en la barra de direcciones, escriba `powershell` y ENTER) y ejecute:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
```

No hace falta “activar” el entorno: todos los comandos de esta guía llaman directamente a `.\.venv\Scripts\python`.
</details>

---

## 4. Probar el sistema SIN celular

Esto es lo mínimo para evaluar el trabajo. Haga **doble clic en `EJECUTAR.bat`**:

```
 1  Ver el sistema SIN camara (abre el notebook en el navegador)
 2  Analizar una foto guardada y ver el resumen en pantalla
 3  Tomar una foto con el celular (DroidCam) y analizarla
 4  Correr las pruebas automaticas
 5  Salir
```

### Opción 2 — Analizar una foto (la más directa)

Escriba **2**, ENTER, y luego el nombre de una foto de `data/imagenes/`. Recomendadas, una por fruta:

| Fruta | Escriba este nombre | Resultado esperado |
|-------|---------------------|--------------------|
| 🍅 Tomate | `foto_20261111_225829_tomate.png` | Tomate 100 %, 5 frutos (1 pintón, 4 maduros) |
| 🍌 Banano | `bananos.jpeg` | Banano 100 %, 4 frutos (1 maduro, 3 sobremaduros) |
| 🥭 Mango | `mango.jpeg` | Mango 100 %, 3 frutos (pintón) |
| 🥭 Mango (otra foto) | `mangos.jpg` | Mango (≈ 57 %), 5 frutos |
| 🍎 Manzana | `manzanas.jpg` | Manzana 100 %, 18 frutos (pintón / madura) |

Se imprime en la terminal:

```
Fruta           : TOMATE (reconocida sola)       <- no hay que elegir la fruta
Frutos contados : 5
Madurez         : Verde 0   Pintón 1   Maduro 4
Diámetro medio  : 6.7 cm
 #  Madurez   Largo x ancho (cm)  Diám. (cm)  Tamaño   <- una fila por fruto
```

y se abre una ventana con los frutos separados y medidos (ciérrela con cualquier tecla). La imagen también queda guardada en `resultados/detecciones/`.

Lo mismo, directo por terminal (sin el menú):

```powershell
.\.venv\Scripts\python herramientas\analizar_foto.py manzanas.jpg
```

### Opción 1 — El notebook completo

Escriba **1**, ENTER. Se abre el navegador con `AgroVision.ipynb` (sin usar cámara). Dentro del notebook: menú **Cell → Run All** (o *Kernel → Restart & Run All*). Tarda ~25 s y muestra el análisis completo con gráficas: segmentación, detección, madurez, tamaño, evaluación y tiempos. Analiza la última foto guardada en `data/imagenes/`.

Para analizar otra foto en el notebook, cambie en la primera celda de código `NOMBRE_IMAGEN = None` por, por ejemplo, `NOMBRE_IMAGEN = "manzanas.jpg"`.

### Opción 4 — Las pruebas automáticas

Corre las 28 pruebas (`tests/`). Deben pasar todas.

---

## 5. Probar CON el celular (DroidCam)

La cámara del celular se transmite por Wi-Fi al PC. **No se necesita ningún tablero de calibración.** Es opcional: todo lo anterior funciona sin celular.

### 5.1 Preparar el celular

1. Instale la app **DroidCam** (App Store en iPhone, Play Store en Android; es gratis).
2. Conecte el celular a la **misma red Wi-Fi que el PC**.
3. Abra la app. En pantalla muestra algo así:

   ```
   WiFi IP:          172.27.29.21
   DroidCam Port:    4747
   ```

   Deje la app **abierta y con la pantalla encendida** mientras trabaja.

### 5.2 Decirle la IP al sistema

La **IP cambia cada vez que cambia de red** (casa, universidad, zona Wi-Fi del celular). Hay que mirar la que muestra la app **en ese momento**.

**Forma fácil (recomendada): guardar la IP en `celular.txt`.** Abra `celular.txt` con el Bloc de notas, deje **solo la IP** en una línea (sin `http://`) y guarde:

```
172.27.29.21
```

Desde ahí, `EJECUTAR.bat` → opción **3** ya usa esa IP (y la muestra antes de abrir la cámara).

### 5.3 El comando de la terminal explicado

El comando completo, con la IP incluida, es:

```powershell
.\.venv\Scripts\python herramientas/tomar_foto.py http://172.27.29.21:4747/video
```

| Parte | Significado |
|-------|-------------|
| `.\.venv\Scripts\python` | El Python del proyecto (el del entorno `.venv` que creó `INSTALAR.bat`) |
| `herramientas/tomar_foto.py` | El programa que abre la cámara, toma la foto y la analiza |
| `http://172.27.29.21:4747/video` | La **dirección del video del celular**: `172.27.29.21` es la **IP** de la app, `4747` el puerto de DroidCam y `/video` la ruta de la imagen |

Con `celular.txt` configurado basta con:

```powershell
.\.venv\Scripts\python herramientas\tomar_foto.py
```

Y estas dos formas son equivalentes (útiles para una IP puntual sin editar el archivo):

```powershell
.\.venv\Scripts\python herramientas\tomar_foto.py 172.27.29.21
.\.venv\Scripts\python herramientas\tomar_foto.py http://172.27.29.21:4747/video
```

**Dónde escribirlo:** en PowerShell **abierto en la carpeta del proyecto** (en el Explorador, dentro de `AgroVision`, clic en la barra de direcciones, escriba `powershell`, ENTER; o en VS Code: *Terminal → Nueva terminal*).

**Prueba rápida de la IP:** escriba `http://172.27.29.21:4747/video` (con su IP) en el navegador del PC. Si se ve el video del celular, la IP está bien. (Ciérrelo después: DroidCam solo admite **un** cliente a la vez.)

### 5.4 Tomar la foto

| Tecla | Acción |
|-------|--------|
| **ESPACIO** | Toma la foto (queda congelada) |
| **g** o **ENTER** | Guardarla |
| **r** | Repetirla |
| **q** | Cancelar |

Al guardar, el sistema **analiza solo**: imprime el resumen en la terminal, muestra la ventana con los frutos medidos (ciérrela con cualquier tecla) y escribe el análisis completo con gráficas en `AgroVision.ipynb`. La foto queda en `data/imagenes/foto_*.png`.

**Para que los centímetros salgan bien:** celular **mirando hacia abajo, a unos 27 cm de la mesa**, fondo oscuro y liso, buena luz, frutos separados o apenas tocándose. Si cambia la altura, los cm dejan de ser exactos (Pequeño/Mediano/Grande siguen valiendo porque son relativos entre los frutos de la foto).

### 5.5 Alternativa sin IP: DroidCam Client (USB o Wi-Fi)

1. Instale *DroidCam Client* en el PC (<https://www.dev47apps.com/>) y pulse **Start**.
2. Use el índice de cámara en vez de la IP:

```powershell
.\.venv\Scripts\python herramientas\probar_camara.py --listar     # ver qué índices hay
.\.venv\Scripts\python herramientas\tomar_foto.py 1               # normalmente el celular es 1
```

---

## 6. Usar el notebook en VS Code (opcional)

1. Instale VS Code y las extensiones **Python** y **Jupyter**.
2. *Archivo → Abrir carpeta* → `AgroVision`; abra `AgroVision.ipynb`.
3. Arriba a la derecha, **Seleccionar kernel → Python Environments →** el que dice **`.venv`** (o “Python (AgroVision)”).
4. **Si no tiene celular**, en la primera celda de código cambie `TOMAR_FOTO = True` por `TOMAR_FOTO = False` (si no, intenta abrir la cámara y espera ~15 s antes de seguir con la última foto; no se rompe, solo se demora).
5. *Ejecutar todo*.

---

## 7. Problemas frecuentes

| Síntoma | Causa y solución |
|---------|------------------|
| `INSTALAR.bat`: **“No se encontró Python”** | Instale Python 3.12 marcando **Add python.exe to PATH** (paso 1) y repita. |
| `python --version` dice *“Python was not found… Microsoft Store”* | Desactive los alias de ejecución de `python.exe` (ver paso 1). |
| Falla `pip install` por internet | Revise la conexión (proxy/firewall de la universidad) y repita `INSTALAR.bat`: continúa donde quedó. |
| `pip` dice **“No such file or directory … Long Path”** | La carpeta está en una ruta muy larga: muévala al Escritorio o a `C:\AgroVision`, borre `.venv` y repita. |
| Su Python es muy nuevo/viejo y falla | Instale Python 3.12, borre la carpeta `.venv` y repita `INSTALAR.bat`. |
| Windows no deja abrir los `.bat` | Clic derecho en el zip → Propiedades → **Desbloquear**; o “Más información → Ejecutar de todas formas”. |
| “No se pudo abrir la fuente 'celular'” / no llega imagen | 1) La app DroidCam debe estar **abierta y en primer plano**. 2) PC y celular en la **misma Wi-Fi**. 3) La IP de `celular.txt` debe ser **la que muestra la app ahora**. 4) Pruebe la URL en el navegador del PC. |
| En la Wi-Fi de la universidad no conecta | Esas redes suelen aislar los dispositivos entre sí. Active la **zona Wi-Fi (hotspot) del celular**, conecte el PC a ella y use la IP que muestre DroidCam. |
| Funciona una vez y luego no | DroidCam acepta **un solo cliente**: cierre `tomar_foto.py`, el navegador con el video y DroidCam Client antes de abrir otra cosa. |
| El firewall de Windows pregunta algo | Permita Python en **redes privadas**. |
| No veo la ventana de la cámara | Búsquela en la barra de tareas; a veces Windows la deja detrás. |
| El notebook no encuentra el kernel (VS Code) | Seleccione el intérprete `.venv`. Con `EJECUTAR.bat` opción 1 no hace falta. |
| VS Code no recarga el notebook tras `tomar_foto.py` | Ciérrelo y ábralo de nuevo. |
| Una prueba falla con otras versiones de librerías | Use exactamente `requirements.txt` en un `.venv` nuevo (versiones fijadas). |

---

## 8. Qué hay en la carpeta

| Ruta | Contenido |
|------|-----------|
| `INSTALAR.bat`, `EJECUTAR.bat` | Instalación y menú con doble clic |
| `celular.txt` | **IP del celular (DroidCam)** — lo único que se edita al cambiar de red |
| `AgroVision.ipynb` | Notebook principal (análisis completo con gráficas) |
| `AgroVision_resultados.html` | El mismo notebook ya ejecutado, para verlo sin instalar |
| `INSTALACION_Y_USO.md` | Esta guía |
| `requirements.txt` | Librerías con versiones exactas |
| `agrovision/` | Código del sistema (segmentación, madurez, reconocimiento de fruta, tamaño, estéreo) |
| `perfiles/` | Un `.json` por fruta: tomate, banano, mango, manzana |
| `herramientas/` | `tomar_foto.py` (cámara), `analizar_foto.py` (foto guardada), `probar_camara.py` |
| `data/imagenes/` | Fotos de prueba y de referencia |
| `modelos/` | Modelo entrenado del reconocedor de fruta |
| `resultados/` | Imágenes y CSV generados por el análisis |
| `tests/` | 28 pruebas automáticas |
| `README.md`, `PROGRESO.md` | Documentación técnica completa y bitácora del proyecto |
