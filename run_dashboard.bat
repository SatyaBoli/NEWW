@echo off
echo.
echo  ========================================
echo   Sentinel Dashboard Launcher
echo  ========================================
echo.
echo  [1] Web Dashboard (HTML - Recommended)
echo  [2] Streamlit Dashboard (Python)
echo.
set /p choice="Select option (1 or 2): "
if "%choice%"=="2" (
    python -m streamlit run app/dashboard.py
) else (
    python app/serve.py
)
