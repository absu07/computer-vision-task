#!/bin/bash

# Start the FastAPI web service in the background
uvicorn app:app --host 0.0.0.0 --port 8080 &

# Wait for a few seconds to ensure the web service starts
sleep 5

# Run the POST request script
python POST_Request.py