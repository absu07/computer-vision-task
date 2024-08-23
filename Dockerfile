# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs/models

# Expose the port (if the application uses one)
EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Copy and set up the script to run the web service and POST request in the background
COPY run_app_and_post.sh /app/
RUN chmod +x /app/run_app_and_post.sh

# Set the entrypoint to Python for the main script
ENTRYPOINT ["python", "main.py"]

# Default command to run the main.py script with default arguments
CMD ["-i", "WIDERFACE_Validation/images/", "-o", "models/", "-g", "WIDERFACE_Validation/labels/"]