@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PYTHON_CMD=py"
) else (
  set "PYTHON_CMD=python"
)
echo ================================================
echo ARIMA-GARCH Simulation and Analysis Lab
 echo ================================================
echo Memasang / memperbarui dependensi (butuh internet)...
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Gagal instal dependensi. Pastikan Python 3.10+ dan internet aktif.
  pause
  exit /b 1
)
echo Menjalankan aplikasi. Tekan CTRL+C untuk menghentikan server.
%PYTHON_CMD% -m streamlit run app.py
pause
