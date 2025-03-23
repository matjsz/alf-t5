FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY alf_t5.py alf_app.py ./
COPY README.md LICENSE ./

# Make directory for models
RUN mkdir -p alf_t5_translator

# Set up volume for models
VOLUME /app/alf_t5_translator

# Expose port if we add a web interface in the future
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["python", "alf_app.py"]
CMD ["--mode", "interactive"] 