@echo off
cd /d "%~dp0"

echo === Dalilak Backend — Push system_prompt.txt update ===
echo.

REM 1. Remove stale git lock
if exist ".git\index.lock" (
    echo Removing stale .git\index.lock...
    del /f /q ".git\index.lock"
)

REM 2. Check git root (backend may share repo with frontend or have its own)
git rev-parse --show-toplevel 2>nul
if errorlevel 1 (
    echo ERROR: Not a git repository. Checking parent directory...
    cd ..
    git rev-parse --show-toplevel 2>nul
    if errorlevel 1 (
        echo ERROR: Could not find git repository. Push manually.
        pause
        exit /b 1
    )
)

REM 3. Stage system_prompt.txt
git add backend/system_prompt.txt
REM Also try relative path in case we're inside backend/
git add system_prompt.txt 2>nul

echo.
echo === Staged files ===
git status --short

REM 4. Commit
git commit -m "feat: system_prompt — add comprehensive Ministry of Finance section (income tax, VAT, built-property tax, indirect taxes, inheritance duty, tax objections, pension dept, customs, land registry)"

REM 5. Push to current branch (whatever is checked out)
echo.
echo === Pushing to GitHub ===
git push origin HEAD

echo.
echo Done. Backend will reload on next Render.com deployment.
pause
