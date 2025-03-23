@echo off
setlocal enabledelayedexpansion

:: Create necessary directories
if not exist alf_t5_translator mkdir alf_t5_translator
if not exist data mkdir data

:: Help function
:show_help
    echo ALF-T5 Docker Runner
    echo --------------------
    echo Usage: docker-run.bat [command]
    echo.
    echo Commands:
    echo   build          - Build the Docker image
    echo   interactive    - Run in interactive mode
    echo   file ^<input^> ^<output^> [direction] [confidence] - Run file translation
    echo   batch          - Run batch translation
    echo   stop           - Stop all running containers
    echo   help           - Show this help message
    echo.
    echo Examples:
    echo   docker-run.bat interactive
    echo   docker-run.bat file data/input.txt data/output.txt c2e yes
    goto :eof

:: Check if command is provided
if "%~1"=="" (
    call :show_help
    exit /b 1
)

:: Process commands
if "%~1"=="build" (
    docker-compose build
    exit /b
)

if "%~1"=="interactive" (
    docker-compose run --rm alf-t5 --mode interactive
    exit /b
)

if "%~1"=="file" (
    if "%~3"=="" (
        echo Error: Missing input or output file
        echo Usage: docker-run.bat file ^<input^> ^<output^> [direction] [confidence]
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
    
    docker-compose run --rm alf-t5 --mode file --input "/app/%INPUT%" --output "/app/%OUTPUT%" --direction "%DIRECTION%" %CONF_FLAG%
    exit /b
)

if "%~1"=="batch" (
    docker-compose run --rm alf-t5 --mode batch
    exit /b
)

if "%~1"=="stop" (
    docker-compose down
    exit /b
)

if "%~1"=="help" (
    call :show_help
    exit /b
)

:: Default case
call :show_help
exit /b 