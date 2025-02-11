# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory to /app inside the container.
WORKDIR /app

# Copy all files from the project directory into the container.
COPY . .

# Install any required system dependencies (e.g., gcc) if needed.
RUN apt-get update && apt-get install -y gcc

# Upgrade pip and install Python dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 (the port our FastAPI app will listen on).
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
