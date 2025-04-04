# Use Python 3.12 as the base image
FROM python:3.12.3

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for PyAudio
RUN apt-get update && apt-get install -y portaudio19-dev

# Copy all project files into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "backend.MasterApi.main:app", "--host", "0.0.0.0", "--port", "8000"]

