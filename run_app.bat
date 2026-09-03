@echo off
setlocal enabledelayedexpansion

title Dogs and Cats Breed Classifier - Web App
echo ===============================================================================
echo            DOGS AND CATS BREED CLASSIFIER - WEB APPLICATION
echo ===============================================================================
echo [INFO] Starting Web Application at %date% %time%
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in system PATH.
    pause
    exit /b 1
)

echo [INFO] Opening Web Server at http://localhost:7860 ...
echo [INFO] Press Ctrl+C in this window to stop the server.
echo.
python app.py --port 7860
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start web application.
    pause
    exit /b %errorlevel%
)
pause
