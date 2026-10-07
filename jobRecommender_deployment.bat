@echo off
setlocal EnableDelayedExpansion

rem ===================================================================
rem  SINGAPORE SPATIAL JOB RADAR - WEB DEPLOYMENT LAUNCHER
rem  Dynamically resolves absolute paths for the current user and location.
rem  Can be copied and executed from anywhere (e.g. Desktop, another folder,
rem  or another user's machine).
rem ===================================================================

echo ===================================================================
echo  Initializing Singapore Spatial Job Radar Web Deployment...
echo ===================================================================

rem -------------------------------------------------------------------
rem 1. RESOLVE ABSOLUTE PATH TO APP DIRECTORY
rem -------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "APP_DIR="

rem Check 1: Is this script inside the project root?
if exist "%SCRIPT_DIR%\13. Deployment_Web_Recommender\server.py" (
    set "APP_DIR=%SCRIPT_DIR%\13. Deployment_Web_Recommender"
) else if exist "%SCRIPT_DIR%\12. Deployment_Web_Recommender\server.py" (
    set "APP_DIR=%SCRIPT_DIR%\12. Deployment_Web_Recommender"
) else if exist "%SCRIPT_DIR%\server.py" (
    set "APP_DIR=%SCRIPT_DIR%"
)

rem Check 2: If script was copied to Desktop or another directory, search known user locations
if not defined APP_DIR (
    for %%P in (
        "%USERPROFILE%\Desktop\Projects\graph"
        "%USERPROFILE%\Desktop\graph"
        "%USERPROFILE%\Projects\graph"
        "%USERPROFILE%\Documents\Projects\graph"
        "%USERPROFILE%\Downloads\graph"
        "C:\Users\Brandon\Desktop\Projects\graph"
        "D:\Projects\graph"
    ) do (
        if not defined APP_DIR (
            if exist "%%~fP\13. Deployment_Web_Recommender\server.py" (
                set "APP_DIR=%%~fP\13. Deployment_Web_Recommender"
            ) else if exist "%%~fP\12. Deployment_Web_Recommender\server.py" (
                set "APP_DIR=%%~fP\12. Deployment_Web_Recommender"
            )
        )
    )
)

if not defined APP_DIR (
    echo.
    echo [ERROR] Could not locate "13. Deployment_Web_Recommender" with "server.py".
    echo Searched in: "%SCRIPT_DIR%" and "%USERPROFILE%\Desktop\Projects\graph".
    echo Please make sure the project directory exists.
    echo.
    pause
    exit /b 1
)

echo [OK] Located Deployment Directory: "%APP_DIR%"

rem -------------------------------------------------------------------
rem 2. RESOLVE & INITIALIZE CONDA
rem -------------------------------------------------------------------
set "CONDA_ACTIVATE="

for %%C in (
    "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    "%USERPROFILE%\Anaconda3\Scripts\activate.bat"
    "%USERPROFILE%\Miniconda3\Scripts\activate.bat"
    "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"
    "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
    "C:\ProgramData\anaconda3\Scripts\activate.bat"
    "C:\ProgramData\miniconda3\Scripts\activate.bat"
    "C:\Users\Brandon\anaconda3\Scripts\activate.bat"
    "C:\anaconda3\Scripts\activate.bat"
    "C:\miniconda3\Scripts\activate.bat"
) do (
    if not defined CONDA_ACTIVATE (
        if exist "%%~fC" (
            set "CONDA_ACTIVATE=%%~fC"
        )
    )
)

if defined CONDA_ACTIVATE (
    echo [OK] Activating Conda via "!CONDA_ACTIVATE!"
    call "!CONDA_ACTIVATE!"
) else (
    echo [INFO] Testing for conda on system PATH...
    where conda >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Conda activate script not found in standard paths. Attempting direct command...
    )
)

rem Activate the specific 'graph' environment
echo [OK] Activating Conda environment 'graph'...
call conda activate graph
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Conda environment 'graph' could not be activated.
    echo Please ensure that Conda is installed and the 'graph' environment exists.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

rem -------------------------------------------------------------------
rem 3. SWITCH TO DEPLOYMENT DIRECTORY & RUN UVICORN
rem -------------------------------------------------------------------
cd /d "%APP_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not switch working directory to "%APP_DIR%".
    pause
    exit /b %ERRORLEVEL%
)

echo ===================================================================
echo  Singapore Spatial Job Radar (GraphSAGE + Leaflet + Ollama)
echo ===================================================================
echo  App Root   : %APP_DIR%
echo  Server URL : http://127.0.0.1:8000
echo ===================================================================
echo.
echo Opening http://127.0.0.1:8000 in your browser...
start http://127.0.0.1:8000

python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload

pause
