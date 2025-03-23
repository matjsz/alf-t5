@echo off
setlocal enabledelayedexpansion

:: Help function
:show_help
    echo ALF-T5 Runner
    echo -------------
    echo Usage: run.bat [command]
    echo.
    echo Commands:
    echo   interactive    - Run in interactive mode
    echo   file ^<input^> ^<output^> [direction] [confidence] - Run file translation
    echo   batch          - Run batch translation
    echo   help           - Show this help message
    echo.
    echo Examples:
    echo   run.bat interactive
    echo   run.bat file input.txt output.txt c2e yes
    goto :eof

:: Check if command is provided
if "%~1"=="" (
    call :show_help
    exit /b 1
)

:: Process commands
if "%~1"=="interactive" (
    python alf_app.py --mode interactive
    exit /b
)

if "%~1"=="file" (
    if "%~3"=="" (
        echo Error: Missing input or output file
        echo Usage: run.bat file ^<input^> ^<output^> [direction] [confidence]
        exit /b 1
    )
    
    set INPUT=%~2
    set OUTPUT=%~3
    
    if "%~4"=="" (
        set DIRECTION=c2e
    ) else (
        set DIRECTION=%~4
    )
    
    set CONF_FLAG=
    if "%~5"=="yes" set CONF_FLAG=--confidence
    if "%~5"=="true" set CONF_FLAG=--confidence
    
    python alf_app.py --mode file --input "%INPUT%" --output "%OUTPUT%" --direction "%DIRECTION%" %CONF_FLAG%
    exit /b
)

if "%~1"=="batch" (
    python alf_app.py --mode batch
    exit /b
)

if "%~1"=="help" (
    call :show_help
    exit /b
)

:: Default case
call :show_help
exit /b 