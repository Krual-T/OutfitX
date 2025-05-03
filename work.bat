@echo off
:: Check if a name is provided
if "%1"=="" (
    echo Please provide a name.
    exit /b 1
)

:: Use PowerShell to get the current date in yyyy-MM-dd format
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set date=%%I

:: Combine the name and date to create the branch name
set branchName=%1/%date%

:: Pull the latest changes from origin main
echo Pulling changes from origin main...
git pull origin main:main

:: Create a new branch based on main with the generated branch name
echo Creating new branch %branchName% based on main...
git checkout -b %branchName% main

echo Done.
pause
