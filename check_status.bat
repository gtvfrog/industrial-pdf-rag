@echo off
echo ========================================
echo   Verificando Status do Sistema
echo ========================================
echo.

echo [1/3] Testando Backend (http://localhost:8000)...
curl -s http://localhost:8000/health
if %errorlevel% neq 0 (
    echo.
    echo ❌ Backend NAO esta rodando!
    echo.
    echo Para iniciar o backend:
    echo    start.bat
    echo.
) else (
    echo.
    echo ✅ Backend OK
)

echo.
echo [2/3] Verificando Streamlit...
tasklist /FI "IMAGENAME eq streamlit.exe" 2>NUL | find /I /N "streamlit.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ✅ Streamlit rodando
) else (
    echo ❌ Streamlit NAO esta rodando
    echo.
    echo Para iniciar Streamlit:
    echo    start_ui.bat
)

echo.
echo [3/3] Verificando pastas...
if exist "documents" (echo ✅ documents/) else (echo ❌ documents/ faltando)
if exist "models_cache" (echo ✅ models_cache/) else (echo ❌ models_cache/ faltando)
if exist "backend\app" (echo ✅ backend/app/) else (echo ❌ backend/app/ faltando)

echo.
echo ========================================
pause
