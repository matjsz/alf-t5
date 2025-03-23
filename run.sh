#!/bin/bash

# Help function
function show_help {
    echo "ALF-T5 Runner"
    echo "-------------"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  interactive    - Run in interactive mode"
    echo "  file <input> <output> [direction] [confidence] - Run file translation"
    echo "  batch          - Run batch translation"
    echo "  help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh interactive"
    echo "  ./run.sh file input.txt output.txt c2e yes"
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Process commands
case "$1" in
    interactive)
        python alf_app.py --mode interactive
        ;;
    file)
        if [ $# -lt 3 ]; then
            echo "Error: Missing input or output file"
            echo "Usage: ./run.sh file <input> <output> [direction] [confidence]"
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
        
        python alf_app.py --mode file --input "$INPUT" --output "$OUTPUT" --direction "$DIRECTION" $CONF_FLAG
        ;;
    batch)
        python alf_app.py --mode batch
        ;;
    help|*)
        show_help
        ;;
esac 