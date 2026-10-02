@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 goto use_python
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto use_python
py -3 -m ai_orchestrate ui %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:use_python
where python >nul 2>&1
if errorlevel 1 goto no_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto old_python
python -m ai_orchestrate ui %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:no_python
echo Python 3.10 or newer was not found. Install Python and try again.
set "EXIT_CODE=1"
goto finish

:old_python
echo Python 3.10 or newer is required. Update Python and try again.
set "EXIT_CODE=1"

:finish
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The ai-orchestrate UI stopped with exit code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
