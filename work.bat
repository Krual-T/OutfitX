@echo off
:: Check if a name is provided
if "%1"=="" (
    echo Please provide a name.
    exit /b 1
)

:: Get the current date in the format yyyy-mm-dd
for /f "tokens=2 delims==" %%I in ('"wmic os get localdatetime /value"') do set datetime=%%I
set year=%datetime:~0,4%
set month=%datetime:~4,2%
set day=%datetime:~6,2%
set date=%year%-%month%-%day%

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
