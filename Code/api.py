#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from logger import DetectionLogger
import uvicorn
import os

# Initialize the FastAPI application
# This 'app' object is what handles all the web requests.
app = FastAPI(
    title="Detection API",
    description="API for retrieving object detection logs from the Raspberry Pi",
    version="1.0.0"
)

# --- Configuration ---
# Define where the database file is located
DB_PATH = "detections.db"

# Initialize our custom logger class.
# This object will handle all the communication with the SQLite database.
logger = DetectionLogger(DB_PATH)

# --- API Endpoints ---

@app.get("/detections")
def get_detections(
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip")
):
    """
    Get a list of past detections.
    
    This endpoint allows you to view the history of what the camera has seen.
    You can use 'limit' and 'offset' to page through results if there are many.
    
    Example: /detections?limit=10&offset=0 (Get the first 10 records)
    """
    # Call the method in our logger class to fetch the data
    return logger.get_all_detections(limit=limit, offset=offset)

@app.get("/detections/{detection_id}")
def get_detection(detection_id: int):
    """
    Get details for a single specific detection.
    
    You must provide the unique ID of the detection you want to see.
    Example: /detections/5
    """
    # Try to find the detection in the database
    detection = logger.get_detection_by_id(detection_id)
    
    # If it doesn't exist, return a 404 "Not Found" error
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")
        
    return detection



@app.get("/stats")
def get_stats():
    """
    Get summary statistics about the system.
    
    Currently returns the total number of detections recorded.
    """
    return logger.get_summary()

@app.get("/images/{filename}")
def get_image(filename: str):
    """
    Retrieve a captured image file.
    
    Args:
        filename (str): The name of the image file (e.g., '2025-12-02_18-30-00.jpg').
    """
    # Define the path to the captures directory
    # We assume it's in the same folder as this script, named 'captures'
    image_path = os.path.join("captures", filename)
    
    # Check if the file exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Return the file directly
    return FileResponse(image_path)

if __name__ == "__main__":
    # This block runs if you execute the script directly (python api.py).
    # It starts the web server using 'uvicorn'.
    # host="0.0.0.0" means it will listen on all network interfaces, 
    # so you can access it from other computers on the network.
    uvicorn.run(app, host="0.0.0.0", port=5000)
