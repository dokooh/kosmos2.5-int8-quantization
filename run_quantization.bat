@echo off
REM Batch script for quantizing Kosmos 2.5 model
REM Usage: run_quantization.bat [approach] [output_dir]

echo Kosmos 2.5 Model Quantization
echo ==============================

REM Activate virtual environment
echo Activating virtual environment...
call kosmos-qint8\Scripts\activate.bat

REM Set default values
set APPROACH=%1
if "%APPROACH%"=="" set APPROACH=avx512_vnni

set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=.\quantized_kosmos

echo.
echo Configuration:
echo   Quantization Approach: %APPROACH%
echo   Output Directory: %OUTPUT_DIR%
echo.

REM Run quantization
echo Starting quantization process...
python quantize_kosmos.py --quantization_approach %APPROACH% --output_dir "%OUTPUT_DIR%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ Quantization completed successfully!
    echo.
    echo To test the quantized model, run:
    echo   python test_quantized_model.py --quantized_model_path "%OUTPUT_DIR%\quantized"
    echo.
) else (
    echo.
    echo ❌ Quantization failed with error code %ERRORLEVEL%
    echo.
)

pause