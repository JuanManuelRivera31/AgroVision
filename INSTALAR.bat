@echo off
rem ============================================================
rem  AgroVision - instalacion con doble clic (Windows)
rem  Crea el entorno .venv, instala las librerias y registra el kernel.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title AgroVision - Instalacion
echo.
echo === AgroVision: instalacion ===
echo.

set "PY="
rem Se prefiere Python 3.12 (con el que se probo); si no esta, 3.13, 3.11, 3.10 o el que haya.
where py >nul 2>nul && (
    for %%V in (3.12 3.13 3.11 3.10) do (
        if not defined PY (
            py -%%V --version >nul 2>nul && set "PY=py -%%V"
        )
    )
    if not defined PY py -3 --version >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>nul && python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] No se encontro Python.
    echo Instale Python 3.12 desde https://www.python.org/downloads/
    echo y marque la casilla "Add python.exe to PATH" en el instalador.
    echo Luego vuelva a hacer doble clic en INSTALAR.bat
    pause
    exit /b 1
)
echo Python encontrado:
%PY% --version

if exist ".venv\Scripts\python.exe" (
    echo El entorno .venv ya existe: se reutiliza.
) else (
    echo Creando entorno virtual .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

echo.
echo Instalando librerias (puede tardar unos minutos, necesita internet) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion de librerias.
    echo Si su Python es muy distinto de 3.12, instale Python 3.12 e intente de nuevo
    echo ^(borre antes la carpeta .venv^).
    pause
    exit /b 1
)

echo.
echo Registrando el kernel de Jupyter ...
".venv\Scripts\python.exe" -m ipykernel install --user --name agrovision --display-name "Python (AgroVision)"

echo.
echo Verificando con las pruebas automaticas ...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 (
    echo [AVISO] Alguna prueba fallo. Revise el mensaje de arriba.
) else (
    echo Todas las pruebas pasaron.
)

echo.
echo ============================================================
echo  Instalacion terminada.
echo  Para abrir el sistema: doble clic en EJECUTAR.bat
echo ============================================================
pause
