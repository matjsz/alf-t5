# ALF-T5 Docker Guide

This guide explains how to use ALF-T5 with Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended)

## Building and Running

### Using Docker Compose (Recommended)

1. Build and start the container:
   ```
   docker-compose up --build
   ```

2. For running in detached mode:
   ```
   docker-compose up -d
   ```

3. To stop the container:
   ```
   docker-compose down
   ```

### Using Docker Directly

1. Build the Docker image:
   ```
   docker build -t alf-t5 .
   ```

2. Run the container in interactive mode:
   ```
   docker run -it --rm -v $(pwd)/alf_t5_translator:/app/alf_t5_translator alf-t5
   ```

## Available Modes

You can run the application in different modes by changing the command:

### Interactive Mode
```
docker-compose run alf-t5 --mode interactive
```

### File Translation Mode
```
docker-compose run alf-t5 --mode file --input /app/data/input.txt --output /app/data/output.txt --direction c2e
```

### Batch Translation Mode
```
docker-compose run alf-t5 --mode batch
```

## With Confidence Scores

To include confidence scores in the output:
```
docker-compose run alf-t5 --mode file --input /app/data/input.txt --output /app/data/output.txt --confidence
```

## Creating a Data Directory

Make sure to create a data directory before running the container with file mode:
```
mkdir -p data
```

Then place your input files in the `data` directory.

## Using Pre-trained Models

To use a pre-trained model, place it in the `alf_t5_translator` directory, and reference it with the `--model` flag:
```
docker-compose run alf-t5 --model alf_t5_translator/your_model --mode interactive
``` 