# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (Required for LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ src/
COPY app/ app/
COPY frontend/ frontend/
COPY models/ models/

# Expose the ports (8000 for API, 8501 for Streamlit)
EXPOSE 8000
EXPOSE 8501

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]