@echo off
REM One-command start for Windows: creates .venv, installs, configures, launches.
setlocal
cd /d "%~dp0"
if not exist .venv (
  echo [riko] creating virtual environment (.venv)
  py -3 -m venv .venv || python -m venv .venv || goto :nopython
  .venv\Scripts\python -m pip install --upgrade pip >nul
  echo [riko] installing Riko Next ^(this can take a few minutes^)
  .venv\Scripts\pip install -e ".[all]" || goto :fail
)
if not exist config.yaml ( copy config.example.yaml config.yaml >nul & echo [riko] created config.yaml )
if not exist .env ( copy .env.example .env >nul & echo [riko] created .env - add API keys there if you use cloud providers )
for /f "usebackq tokens=1,* delims==" %%A in (".env") do if /i "%%A"=="GPT_SOVITS_DIR" set "GPT_SOVITS_DIR=%%B"
if defined GPT_SOVITS_DIR if exist "%GPT_SOVITS_DIR%\api_v2.py" (
  echo [riko] starting GPT-SoVITS API from %GPT_SOVITS_DIR%
  if exist "%GPT_SOVITS_DIR%\runtime\python.exe" ( start "GPT-SoVITS" /D "%GPT_SOVITS_DIR%" runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml ) else ( start "GPT-SoVITS" /D "%GPT_SOVITS_DIR%" python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml )
)
.venv\Scripts\python -m riko serve --open %*
goto :eof
:nopython
echo Python 3.11+ was not found. Install it from https://www.python.org/downloads/ (tick "Add to PATH").
pause
goto :eof
:fail
echo Installation failed - see the messages above.
pause
