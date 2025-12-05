# Edge-AI Wildlife Detection and Roadside Warning System with Logging and API

## 1. Summary
This project implements a real-time **Edge-AI Wildlife Detection and Roadside Warning System** on a Raspberry Pi 5, leveraging the **Coral Edge TPU** for high-performance inference. The system uses a **custom YOLO model trained in Google Colab** to detect specific wildlife (e.g., deer, moose) in video streams. Upon detection, it triggers a hardware warning (Beacon light (LED light used in prototype)), logs the event to a persistent database, and exposes the data via a RESTful API. This solution demonstrates the effective integration of custom AI models, edge computing, and IoT for enhancing roadside safety.

## 2. System Architecture
![System architecture diagram](<System Architecture.png>)

The system is composed of three main modules:
1.  **Detection Module (`detect.py`)**: The core application that captures video, runs inference, and manages hardware I/O.
2.  **Logging Module (`logger.py`)**: A dedicated handler for SQLite database operations.
3.  **API Module (`api.py`)**: A FastAPI-based web server for data retrieval.

### Hardware Stack
-   **Platform**: Raspberry Pi 5
-   **Accelerator**: Coral USB Accelerator (Edge TPU)
-   **Camera**: USB Webcam or Pi Camera
-   **Output**: GPIO-connected LED

### Software Stack
-   **Language**: Python 3
-   **AI Framework**: TensorFlow Lite (Edge TPU runtime)
-   **Web Framework**: FastAPI
-   **Database**: SQLite
-   **Computer Vision**: OpenCV

## 3. Implementation Details

### 3.1 Object Detection (`detect.py`)
The detection script utilizes a multi-threaded architecture to maximize performance:
-   **Capture Thread**: Continuously reads frames from the camera to ensure the inference engine always has the latest image.
-   **Inference Thread**: Preprocesses images (letterboxing, quantization), invokes the custom YOLO model (trained in Colab) on the Edge TPU, and post-processes outputs (NMS). It also handles the logic for triggering the LED and calling the logger.
-   **LED Thread**: Manages the non-blocking flashing of the LED to indicate detection events.
-   **Main Thread**: Handles the GUI display using OpenCV, drawing bounding boxes and frame rate statistics.

### 3.2 Data Persistence (`logger.py`)
To ensure data integrity and ease of access, an SQLite database is used. The `DetectionLogger` class encapsulates all SQL operations.
-   **Schema**: The `detections` table stores timestamps, device IDs, image paths, and JSON-serialized detection details (bounding boxes, confidence scores).
-   **Features**: Automatic table creation, JSON serialization/deserialization, and pagination support.

### 3.3 RESTful API (`api.py`)
A modern API was built using **FastAPI** to allow remote monitoring and data analysis.
-   **Endpoints**:
    -   `GET /detections`: Retrieves a paginated list of detection events.
    -   `GET /detections/{id}`: detailed view of a specific event.
    -   `GET /stats`: Provides summary metrics (e.g., total detection count).
    -   `GET /docs`: Auto-generated interactive documentation (Swagger UI).

## 4. Challenges and Solutions

### 4.1 Concurrency and Thread Safety
**Challenge**: Sharing data (frames, detection lists) between threads caused race conditions.
**Solution**: Implemented `threading.Lock()` for all shared resources (`frame_lock`, `det_lock`, `led_lock`) to ensure atomic updates.

### 4.2 Edge TPU Integration
**Challenge**: The Edge TPU requires specific quantization (int8) and input preprocessing.
**Solution**: Implemented a robust `preprocess` function to handle resizing, padding (letterboxing), and normalization to match the model's input tensor requirements.

### 4.3 Raspberry Pi 5 Compatibility
**Challenge**: The Pi 5 uses a new GPIO chip structure and Wayland display server, causing issues with standard libraries.
**Solution**:
-   Used `gpiozero` with the `lgpio` pin factory for reliable hardware control.
-   Configured environment variables (`QT_QPA_PLATFORM=xcb`) to ensure OpenCV windows render correctly on the desktop.

## 5. Conclusion
The project successfully delivers a robust Edge AI solution tailored for wildlife detection and roadside safety. By leveraging a custom YOLO model trained specifically for this application in Google Colab, the system achieves high accuracy in identifying target species. It operates in real-time with low latency on the Raspberry Pi 5, reliably logs detection events for historical analysis, and provides a modern API for data consumption. This architecture demonstrates the practical viability of low-cost, high-performance edge computing for environmental monitoring and safety systems.

## 6. Future Work
-   **Web Dashboard**: Create a frontend to visualize the API data.
-   **Cloud Sync**: Automatically upload captured images to cloud storage.
-   **Alerting**: Send email or SMS notifications upon detection.
