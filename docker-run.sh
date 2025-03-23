#!/bin/bash

# Create necessary directories
mkdir -p alf_t5_translator
mkdir -p data

# Help function
function show_help {
    echo "ALF-T5 Docker Runner"
    echo "--------------------"
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build          - Build the Docker image"
    echo "  interactive    - Run in interactive mode"
    echo "  file <input> <output> [direction] [confidence] - Run file translation"
    echo "  batch          - Run batch translation"
    echo "  stop           - Stop all running containers"
    echo "  help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./docker-run.sh interactive"
    echo "  ./docker-run.sh file data/input.txt data/output.txt c2e yes"
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Process commands
case "$1" in
    build)
        docker-compose build
        ;;
    interactive)
        docker-compose run --rm alf-t5 --mode interactive
        ;;
    file)
        if [ $# -lt 3 ]; then
            echo "Error: Missing input or output file"
            echo "Usage: ./docker-run.sh file <input> <output> [direction] [confidence]"
            exit 1
        fi
        
        INPUT="$2"
        OUTPUT="$3"
        DIRECTION=${4:-c2e}
        CONFIDENCE=$5
        
        CONF_FLAG=""
        if [ "$CONFIDENCE" = "yes" ] || [ "$CONFIDENCE" = "true" ]; then
            CONF_FLAG="--confidence"
        fi
        
        docker-compose run --rm alf-t5 --mode file --input "/app/$INPUT" --output "/app/$OUTPUT" --direction "$DIRECTION" $CONF_FLAG
        ;;
    batch)
        docker-compose run --rm alf-t5 --mode batch
        ;;
    stop)
        docker-compose down
        ;;
    help|*)
        show_help
        ;;
esac 