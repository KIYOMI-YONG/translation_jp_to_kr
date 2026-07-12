@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo 번역기를 실행합니다.
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [오류] .venv 가상환경을 찾을 수 없습니다.
    echo 현재 폴더에 .venv가 있는지 확인하세요.
    echo.
    pause
    exit /b 1
)

if not exist "trans.py" (
    echo [오류] trans.py 파일을 찾을 수 없습니다.
    echo 배치 파일과 trans.py를 같은 폴더에 두세요.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "trans.py"

set EXIT_CODE=%errorlevel%

echo.
echo 번역기 실행이 종료되었습니다.
echo 종료 코드: %EXIT_CODE%
echo.

pause
exit /b %EXIT_CODE%