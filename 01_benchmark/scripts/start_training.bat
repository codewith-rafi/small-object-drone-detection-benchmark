@echo off
REM Batch file to start RTX 5060 drone detection training
REM Run in conda environment: drone_detection

echo ========================================
echo RTX 5060 Drone Detection Training
echo ========================================
echo.

echo Step 1: Validation (5 minutes)
echo conda run -n drone_detection python validate_comprehensive.py
echo.

echo Step 2: GPU Monitoring (separate terminal)
echo nvidia-smi dmon -s mu -d 5
echo.

echo Step 3: Batch Discovery (2-3 hours)
echo conda run -n drone_detection python discover_batch_final.py
echo.

echo Step 4: Training (sequential)
echo conda run -n drone_detection python train_yolov8_xl_final.py
echo conda run -n drone_detection python train_yolo11x_final.py
echo conda run -n drone_detection python train_yolov10_x_final.py
echo.

echo Step 5: SAHI Evaluation
echo conda run -n drone_detection python sahi_evaluation_final.py
echo.

echo Total estimated time: 3-4 days
echo ========================================
pause