@echo off
rem Initialize Conda's base environment
call "C:\Users\Brandon\anaconda3\Scripts\activate.bat" "C:\Users\Brandon\anaconda3"
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Conda base environment could not be initialized.
    echo Please ensure the path "C:\Users\Brandon\anaconda3" is correct.
    pause
    exit /b %ERRORLEVEL%
)

rem Activate your specific Conda environment
call conda activate graph
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Conda environment 'graph' could not be activated.
    echo Please ensure the environment name is correct and it exists.
    pause
    exit /b %ERRORLEVEL%
)

cd "%~dp0"
echo ===================================================================
echo Starting Singapore Spatial Job Radar (FastAPI + Modern Web App)...
echo Opening http://localhost:8000 in your browser...
echo ===================================================================

start http://localhost:8000
python -m uvicorn server:app --host 127.0.0.1 --port 8000

pause
