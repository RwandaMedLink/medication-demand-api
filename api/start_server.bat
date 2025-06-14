@echo off
echo.
echo ======================================
echo   Rwanda MedLink API - Quick Start
echo ======================================
echo.

REM Check if we're in the right directory
if not exist "main.py" (
    echo Error: main.py not found. Please run this script from the api directory.
    echo Expected location: c:\Users\user\dev\rwanda_medlink_model\api\
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python or add it to PATH.
    pause
    exit /b 1
)
echo ✅ Python found

echo.
echo [2/3] Testing API imports...
python -c "import sys; sys.path.insert(0, '.'); from app_factory import create_app; print('✅ API imports successful')" 2>nul
if errorlevel 1 (
    echo ❌ Import test failed. Please check dependencies.
    echo Try: pip install flask pandas numpy scikit-learn joblib
    pause
    exit /b 1
)

echo.
echo [3/3] Starting development server...
echo.
echo 🚀 Rwanda MedLink API Server Starting...
echo 📍 URL: http://localhost:5000
echo 🌐 Web Interface: http://localhost:5000/api/web/predict
echo 📋 Health Check: http://localhost:5000/health
echo.
echo ⏹️ Press Ctrl+C to stop the server
echo ======================================
echo.

REM Start the Flask development server
python main.py

echo.
echo Server stopped.
pause
