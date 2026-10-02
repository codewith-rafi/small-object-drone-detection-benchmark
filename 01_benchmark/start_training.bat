@echo off
echo ========================================
echo DRONE DETECTION BENCHMARK - TRAINING
echo ========================================
echo.

REM Activate conda environment
call conda activate drone_detection

if errorlevel 1 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)

echo Environment: drone_detection
echo Python: %CONDA_PREFIX%\python.exe
echo.

REM Check GPU
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ========================================
echo OPTIONS:
echo ========================================
echo 1. Train YOLOv8-XL Baseline
echo 2. Monitor Training Progress
echo 3. Run EDA (Exploratory Data Analysis)
echo 4. Test Installations
echo 5. Exit
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting YOLOv8-XL baseline training...
    echo This will take several hours. Check runs/yolov8_xl_baseline for progress.
    echo.
    python scripts/train_yolov8_xl.py
) else if "%choice%"=="2" (
    echo.
    echo Monitoring training progress...
    python scripts/monitor_training.py
) else if "%choice%"=="3" (
    echo.
    echo Running Exploratory Data Analysis...
    python scripts/eda_visdrone.py
) else if "%choice%"=="4" (
    echo.
    echo Testing installations...
    python scripts/test_installations.py
) else if "%choice%"=="5" (
    echo.
    echo Exiting...
) else (
    echo.
    echo Invalid choice
)

echo.
pause