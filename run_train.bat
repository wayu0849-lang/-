@echo off
setlocal enabledelayedexpansion

title Dogs and Cats Classifier - Model Training & Evaluation
echo ===============================================================================
echo        DOGS AND CATS BREED CLASSIFIER - TRAINING & EVALUATION PIPELINE
echo ===============================================================================
echo [INFO] Starting training pipeline execution at %date% %time%
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in system PATH.
    pause
    exit /b 1
)

echo [STEP 1/3] Training Deep Learning Model (Transfer Learning)...
python src/train.py --model_name mobilenetv3_large_100 --epochs 5 --batch_size 32
if %errorlevel% neq 0 (
    echo [ERROR] Model training failed.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Training completed. Best model checkpoint saved to models/best_model.pth
echo.

echo [STEP 2/3] Evaluating Best Model on Held-out Test Set...
python src/evaluate.py --checkpoint models/best_model.pth --reports_dir reports/evaluation
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation failed.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Evaluation completed. Reports generated at reports/evaluation/
echo.

echo [STEP 3/3] Running Inference Demo on a Sample Test Image...
for /f "delims=" %%i in ('powershell -Command "Get-ChildItem -Path data/test -Recurse -Filter *.jpg | Select-Object -First 1 -ExpandProperty FullName"') do set "SAMPLE_IMG=%%i"
if defined SAMPLE_IMG (
    echo [INFO] Testing image: !SAMPLE_IMG!
    python src/predict.py --image "!SAMPLE_IMG!"
) else (
    echo [INFO] No sample image found in data/test for quick demo.
)

echo ===============================================================================
echo                    TRAINING & EVALUATION FINISHED SUCCESSFULLY
echo ===============================================================================
echo Generated Artifacts:
echo   - Best Model Checkpoint: models/best_model.pth
echo   - Class Mapping:         models/class_mapping.json
echo   - Training Curves:       reports/training/training_curves.png
echo   - Training Report:       reports/training/training_report.md
echo   - Confusion Matrix:      reports/evaluation/confusion_matrix.png
echo   - Error Analysis:        reports/evaluation/error_analysis.png
echo   - Test Eval Report:      reports/evaluation/test_evaluation_report.md
echo ===============================================================================
echo.
pause
