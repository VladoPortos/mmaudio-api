FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone MMAudio repository
RUN git clone https://github.com/hkchengrex/MMAudio.git /app/MMAudio
WORKDIR /app/MMAudio
RUN pip install -e .
WORKDIR /app

# Copy application code first (excluding MMAudio if it exists locally)
COPY api/ /app/api/
COPY main.py requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
