@echo off
setlocal enabledelayedexpansion

title Dogs and Cats Classifier - End-to-End ML Pipeline
echo ===============================================================================
echo            DOGS AND CATS CLASSIFIER - END-TO-END DATA PIPELINE
echo ===============================================================================
echo [INFO] Starting pipeline execution at %date% %time%
echo.

:: Ensure MinGit / Git is accessible if installed in local AppData
if exist "%LOCALAPPDATA%\Programs\Git\cmd" (
    set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"
)

:: Check Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in system PATH. Please install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo [STEP 1/5] Installing and verifying Python dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Dependencies installed successfully.
echo.

echo [STEP 2/5] Downloading dataset from Kaggle via KaggleHub...
python src/download_data.py
if %errorlevel% neq 0 (
    echo [ERROR] Data download step failed.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Dataset downloaded and verified.
echo.

echo [STEP 3/5] Running Exploratory Data Analysis (EDA)...
python src/eda.py
if %errorlevel% neq 0 (
    echo [ERROR] EDA step failed.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] EDA completed. Report generated at reports/eda/eda_report.md
echo.

echo [STEP 4/5] Running Image Preprocessing & Augmentation Pipeline...
python src/preprocess.py
if %errorlevel% neq 0 (
    echo [ERROR] Image preprocessing step failed.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Preprocessing completed. Report generated at reports/preprocessing/preprocessing_report.md
echo.

echo [STEP 5/6] Running Data Splitting Verification...
python src/split_data.py
if %errorlevel% neq 0 (
    echo [ERROR] Data splitting step failed.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Data splitting completed. Report generated at reports/data_splitting/data_splitting_report.md
echo.

echo [STEP 6/6] Generating PowerPoint Presentation Slides...
python src/generate_slides.py
if %errorlevel% neq 0 (
    echo [WARNING] Presentation generation encountered an issue, skipping.
) else (
    echo [SUCCESS] Presentation slides generated at presentation_dogs_cats_pipeline.pptx
)
echo.

echo ===============================================================================
echo                    DATA PIPELINE FINISHED SUCCESSFULLY
echo ===============================================================================
echo Summary of Generated Reports and Artifacts:
echo   - Presentation Slides:   presentation_dogs_cats_pipeline.pptx
echo   - Speaker Script Notes:  reports/presentation_slides.md
echo   - Master Report:         reports/project_summary_report.md
echo   - EDA Report:            reports/eda/eda_report.md
echo   - Preprocessing Report:  reports/preprocessing/preprocessing_report.md
echo   - Data Splitting Report: reports/data_splitting/data_splitting_report.md
echo ===============================================================================
echo [TIP] To train the Deep Learning model, run 'run_train.bat' or:
echo       python src/train.py --epochs 10 --batch_size 32
echo ===============================================================================
echo.
pause
