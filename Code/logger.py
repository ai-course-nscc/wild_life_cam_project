#!/usr/bin/env python3
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

class DetectionLogger:
    """
    A class to handle all database interactions for object detection.
    It manages the SQLite database, creates tables, and handles saving/retrieving data.
    """
    
    def __init__(self, db_path="detections.db"):
        """
        Initialize the logger.
        
        Args:
            db_path (str): The file path where the SQLite database will be stored.
                           Defaults to "detections.db" in the current directory.
        """
        self.db_path = db_path
        # Ensure the table exists as soon as the logger is created
        self._create_table()

    def _create_table(self):
        """
        Creates the 'detections' table in the database if it doesn't already exist.
        This table holds all the information about each detection event.
        """
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Execute SQL command to create the table
        # We use IF NOT EXISTS so this doesn't fail if the table is already there.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique ID for each record
                timestamp TEXT,                        -- When the detection happened
                device_id TEXT,                        -- ID of the Raspberry Pi
                detected BOOLEAN,                      -- True if object found, else False
                night_mode BOOLEAN,                    -- True if night mode was active
                image_path TEXT,                       -- Path to the saved image file
                detections TEXT,                       -- JSON string of all objects found
                species_count TEXT,                    -- JSON string counting each species
                summary TEXT                           -- A human-readable summary string
            )
        ''')
        
        # Save changes and close the connection
        conn.commit()
        conn.close()

    def log_detection(self, data):
        """
        Saves a single detection event to the database.
        
        Args:
            data (dict): A dictionary containing all the detection details.
                         It has keys like 'timestamp', 'device_id', etc.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # The database expects text (strings) for complex data like lists or dictionaries.
        # We use json.dumps() to convert Python lists/dicts into JSON strings.
        detections_json = json.dumps(data.get("detections", []))
        species_count_json = json.dumps(data.get("species_count", {}))

        # Insert the data into the table.
        # We use '?' placeholders to prevent SQL injection attacks and handle formatting.
        cursor.execute('''
            INSERT INTO detections (
                timestamp, device_id, detected, night_mode, image_path, 
                detections, species_count, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get("timestamp"),
            data.get("device_id"),
            data.get("detected"),
            data.get("night_mode"),
            data.get("image_path"),
            detections_json,
            species_count_json,
            data.get("summary")
        ))

        conn.commit()
        conn.close()
        print(f"[Logger] Logged detection event at {data.get('timestamp')}")

    def get_all_detections(self, limit=100, offset=0):
        """
        Retrieves a list of past detections from the database.
        
        Args:
            limit (int): The maximum number of records to return (default 100).
            offset (int): How many records to skip (useful for pagination).
            
        Returns:
            list: A list of dictionaries, where each dictionary is one detection event.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This allows us to access columns by name
        cursor = conn.cursor()
        
        # Get records sorted by newest first (ORDER BY id DESC)
        cursor.execute('''
            SELECT * FROM detections 
            ORDER BY id DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            # Convert the database row to a standard Python dictionary
            d = dict(row)
            
            # Convert the JSON strings back into Python lists/dictionaries
            try:
                d['detections'] = json.loads(d['detections'])
                d['species_count'] = json.loads(d['species_count'])
            except (json.JSONDecodeError, TypeError):
                # If something goes wrong, just keep them as strings
                pass 
            results.append(d)
            
        return results

    def get_detection_by_id(self, detection_id):
        """
        Retrieves a single specific detection by its unique ID.
        
        Args:
            detection_id (int): The ID of the detection to find.
            
        Returns:
            dict: The detection data, or None if no record was found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            d = dict(row)
            # Convert JSON strings back to Python objects
            try:
                d['detections'] = json.loads(d['detections'])
                d['species_count'] = json.loads(d['species_count'])
            except (json.JSONDecodeError, TypeError):
                pass
            return d
        return None

    def get_summary(self):
        """
        Calculates basic statistics about the stored data.
        
        Returns:
            dict: A dictionary with summary stats (e.g., total number of detections).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count how many rows are in the table
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_detections": total_count
        }

if __name__ == "__main__":
    # This block only runs if you execute this script directly (e.g., python logger.py).
    # It's useful for testing the class without running the full detection system.
    print("Testing DetectionLogger...")
    logger = DetectionLogger("test_detections.db")
    
    sample_data = {
        "timestamp": datetime.now().isoformat(),
        "device_id": "unit-test",
        "detected": True,
        "night_mode": False,
        "image_path": "captures/test.jpg",
        "detections": [{"species": "deer", "confidence": 0.9}],
        "species_count": {"deer": 1},
        "summary": "Test detection"
    }
    
    logger.log_detection(sample_data)
    print("Test complete. Data logged to test_detections.db")
