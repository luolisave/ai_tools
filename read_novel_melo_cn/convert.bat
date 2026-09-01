@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ======================================
echo Step 1/3: Clearing novel and novel_mp3
echo ======================================
if exist "novel\*" (del /f /q /s "novel\*" 2>nul)
if exist "novel_mp3\*" (del /f /q /s "novel_mp3\*" 2>nul)

echo.
echo ======================================
echo Step 2/3: Running split_novel.py
echo ======================================
py -3.11 .\split_novel.py
if errorlevel 1 (
    echo [ERROR] split_novel.py failed.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Step 3/3: Running tts4.py
echo ======================================
py -3.11 .\tts4.py
if errorlevel 1 (
    echo [ERROR] tts4.py failed.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Done! All steps completed successfully.
echo ======================================
pause
endlocal