@echo off
rem ============================================================
rem  AgroVision - menu de ejecucion (doble clic)
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title AgroVision
if not exist ".venv\Scripts\python.exe" (
    echo Primero haga doble clic en INSTALAR.bat
    pause
    exit /b 1
)
set "PYTHONIOENCODING=utf-8"
rem Fuerza a Jupyter a usar el Python de ESTE proyecto (y no otro entorno activo en el PC)
set "VIRTUAL_ENV=%~dp0.venv"

:menu
echo.
echo ============================================================
echo   AgroVision - que desea hacer?
echo ============================================================
echo   1  Ver el sistema SIN camara (abre el notebook en el navegador)
echo   2  Analizar una foto guardada y ver el resumen en pantalla
echo   3  Tomar una foto con el celular (DroidCam) y analizarla
echo   4  Correr las pruebas automaticas
echo   5  Salir
echo.
set /p op="Escriba el numero y ENTER: "

if "%op%"=="1" goto notebook
if "%op%"=="2" goto foto
if "%op%"=="3" goto camara
if "%op%"=="4" goto pruebas
if "%op%"=="5" exit /b 0
goto menu

:notebook
rem Sin camara: el notebook analiza la ultima foto guardada en data\imagenes
set "AGROVISION_SIN_CAMARA=1"
echo Abriendo el notebook... (cierre esta ventana negra para terminar)
".venv\Scripts\python.exe" -m notebook AgroVision.ipynb
goto menu

:foto
echo.
echo Fotos disponibles en data\imagenes:
dir /b "data\imagenes"
echo.
set /p nombre="Nombre de la foto (ej. mangos.jpg): "
".venv\Scripts\python.exe" herramientas\analizar_foto.py "data\imagenes\%nombre%"
pause
goto menu

:camara
echo.
echo Celular configurado (celular.txt):
".venv\Scripts\python.exe" -c "from agrovision import config as c; print(' ', c.ORIGEN_CELULAR)"
echo Si la IP de la app DroidCam es otra, cierre y edite celular.txt.
echo ESPACIO = tomar foto, g = guardar, r = repetir, q = cancelar
echo.
".venv\Scripts\python.exe" herramientas\tomar_foto.py
pause
goto menu

:pruebas
".venv\Scripts\python.exe" -m pytest -q
pause
goto menu
