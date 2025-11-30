@echo off
echo Parando processos anteriores do Streamlit...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Iniciando Streamlit UI...
streamlit run frontend/Chat.py
