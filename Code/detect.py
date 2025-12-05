#!/usr/bin/env python3
import os
import time
import threading
from pathlib import Path

# --- Qt / OpenCV GUI setup (fix Wayland issues) ---------------------------
# These settings help OpenCV display windows correctly on the Raspberry Pi 5 desktop environment.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"  # Required for Raspberry Pi 5 GPIO control

import cv2
import numpy as np
from tflite_runtime import interpreter as tflite
from gpiozero import LED
from datetime import datetime
from logger import DetectionLogger

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# Path to the AI model file. This model is optimized for the Coral Edge TPU.
MODEL_PATH = Path("yolo11n_int8_edgetpu.tflite")

# Confidence threshold: detections below this score (0.0 to 1.0) are ignored.
CONF_THRESH = 0.4

# NMS (Non-Maximum Suppression) threshold: helps remove duplicate boxes for the same object.
NMS_IOU_THRESH = 0.5

# Camera index: 0 usually refers to the first connected camera (e.g., USB or Pi Camera).
CAM_INDEX = 0

# List of object classes the model can detect.
# We are currently only interested in "deer" (class index 0).
COCO_NAMES = [
    "deer",
]

# LED config
LED_PIN = 18          # The GPIO pin number where the LED is connected
FLASH_DURATION = 5.0  # How long (in seconds) the LED keeps flashing after a detection

# Logging config
DB_PATH = "detections.db"       # Database file name
CAPTURE_DIR = Path("captures")  # Folder to save detection images
CAPTURE_DIR.mkdir(exist_ok=True) # Create the folder if it doesn't exist
LOG_COOLDOWN = 1.0              # Minimum seconds between saving logs (prevents spamming)
DEVICE_ID = "unit-001"          # A unique name for this device

# ------------------------------------------------------------
# Shared state (between threads)
# ------------------------------------------------------------
# These variables are shared between the different threads (Capture, Inference, LED, Main).
latest_frame = None          # Stores the most recent image from the camera
latest_detections = []       # Stores the most recent list of detected objects

# Locks prevent two threads from modifying the same variable at the exact same time.
frame_lock = threading.Lock()
det_lock = threading.Lock()
stop_flag = False            # Set to True to tell all threads to stop running

# FPS (Frames Per Second) tracking variables
infer_fps = 0.0
camera_fps = 0.0

infer_lock = threading.Lock()
camera_lock = threading.Lock()

# LED shared state
led = None
last_detection_time = 0.0    # Timestamp of the last time a deer was seen
led_flashing = False         # Is the LED currently in blinking mode?
led_lock = threading.Lock()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def letterbox_resize(img, new_shape):
    """
    Resizes an image to fit the model's expected size while keeping the aspect ratio.
    It adds gray padding (letterboxing) to fill the empty space.
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (resized_w, resized_h))

    # Create a gray canvas
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    
    # Calculate where to paste the resized image to center it
    dw = (new_w - resized_w) // 2
    dh = (new_h - resized_h) // 2
    canvas[dh:dh + resized_h, dw:dw + resized_w, :] = resized

    return canvas, scale, dw, dh


def preprocess(frame, input_details):
    """
    Prepares the camera frame for the AI model.
    1. Resizes/Letterboxes the image.
    2. Normalizes pixel values.
    3. Quantizes values (converts to int8) because the Edge TPU model expects integers.
    """
    _, h, w, _ = input_details[0]["shape"]
    # letterbox to model size
    img_lb, scale, dw, dh = letterbox_resize(frame, (h, w))

    img = img_lb.astype(np.float32) / 255.0

    scale_q, zero_point_q = input_details[0]["quantization"]
    if scale_q == 0:
        raise ValueError("Input quantization scale is zero.")

    # Apply quantization math
    img_q = img / scale_q + zero_point_q
    img_q = np.clip(img_q, -128, 127).astype(np.int8)
    img_q = np.expand_dims(img_q, axis=0)  # Add batch dimension: [1, H, W, 3]
    return img_q, scale, dw, dh


def nms_boxes(boxes, scores, iou_thresh):
    """
    Performs Non-Maximum Suppression (NMS).
    If multiple boxes overlap significantly on the same object, this keeps only the best one.
    """
    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1] # Sort boxes by confidence score (highest first)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate intersection with other boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        # Keep boxes that don't overlap too much
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def decode_yolo_output(output, output_details, orig_shape, scale, dw, dh, input_size):
    """
    Decode YOLOv8-style output from EdgeTPU model.
    output shape: [1, 84, 8400] (int8 quantized).
    Returns list of [score, class_id, x1, y1, x2, y2] in original frame coords.
    """
    out = output[0]  # [84, 8400]
    s, zp = output_details[0]["quantization"]

    out = out.astype(np.float32)
    out = (out - zp) * s  # dequantize

    boxes = out[0:4, :]      # Box coordinates
    scores_all = out[4:, :]  # Class scores

    class_ids = np.argmax(scores_all, axis=0)
    class_scores = scores_all[class_ids, np.arange(scores_all.shape[1])]

    # Filter out weak detections
    mask = class_scores > CONF_THRESH
    if not np.any(mask):
        return []

    class_ids = class_ids[mask]
    class_scores = class_scores[mask]
    boxes = boxes[:, mask]  # [4, N]

    ih, iw = input_size  # model input H, W
    oh, ow = orig_shape  # original frame H, W

    # YOLO boxes are (cx, cy, w, h) normalized 0..1
    cx, cy, w, h = boxes
    # Convert to xyxy in letterboxed image coords
    x1 = (cx - w / 2) * iw
    y1 = (cy - h / 2) * ih
    x2 = (cx + w / 2) * iw
    y2 = (cy + h / 2) * ih

    # Undo letterbox: shift and scale back to original image coordinates
    x1 = (x1 - dw) / (iw - 2 * dw + 1e-6) * ow
    x2 = (x2 - dw) / (iw - 2 * dw + 1e-6) * ow
    y1 = (y1 - dh) / (ih - 2 * dh + 1e-6) * oh
    y2 = (y2 - dh) / (ih - 2 * dh + 1e-6) * oh

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # [N,4]

    # ---- Class-aware NMS ----
    final_dets = []
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        cls_mask = class_ids == cls
        if not np.any(cls_mask):
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = class_scores[cls_mask]

        keep = nms_boxes(cls_boxes, cls_scores, NMS_IOU_THRESH)
        for idx in keep:
            b = cls_boxes[idx]
            s = float(cls_scores[idx])
            final_dets.append(
                [s, int(cls), float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            )

    final_dets.sort(key=lambda d: d[0], reverse=True)
    return final_dets


def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the image frame for visualization.
    """
    for score, cls, x1, y1, x2, y2 in detections[:50]:
        cls_name = COCO_NAMES[cls] if cls < len(COCO_NAMES) else f"class_{cls}"
        color = (0, 255, 0) # Green color
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        # Draw the rectangle
        cv2.rectangle(frame, p1, p2, color, 1)

        # Draw the label text
        label = f"{cls_name} {score:.2f}"
        cv2.putText(
            frame, label, (p1[0], max(p1[1] - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
    return frame


# ------------------------------------------------------------
# LED manager thread
# ------------------------------------------------------------
def led_manager_thread_func():
    """
    Manages LED flashing based on last_detection_time.
    It runs in a separate thread so it doesn't block the camera or AI.

    - When a deer is detected, inference thread sets last_detection_time
      and starts led.blink() if not already flashing.
    - As long as detections keep happening, last_detection_time is refreshed.
    - If no detection happens for FLASH_DURATION seconds, we stop the blink
      and turn the LED off.
    """
    global stop_flag, led_flashing, last_detection_time, led

    BLINK_ON_TIME = 0.1   # seconds LED is on each blink
    BLINK_OFF_TIME = 0.1  # seconds LED is off each blink
    print("[LED] Manager thread started.")

    while not stop_flag:
        now = time.time()
        with led_lock:
            if led_flashing:
                # If we've been detection-free for FLASH_DURATION, stop flashing
                if now - last_detection_time > FLASH_DURATION:
                    if led is not None:
                        led.off()
                    led_flashing = False
        time.sleep(0.05)  # small sleep to avoid busy loop

    # Ensure LED is off when the program exits
    if led is not None:
        led.off()
    print("[LED] Manager thread stopped.")


# ------------------------------------------------------------
# Threads
# ------------------------------------------------------------
def capture_thread_func():
    """
    Continuously reads frames from the camera.
    This ensures we always have the freshest image ready for processing.
    """
    global latest_frame, stop_flag, camera_fps

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Capture] Could not open camera index {CAM_INDEX}")
        stop_flag = True
        return

    # Set resolution (optional, helps with speed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap_count = 0
    cap_prev_time = time.time()

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("[Capture] Failed to grab frame.")
            time.sleep(0.01)
            continue

        # Safely update the shared 'latest_frame' variable
        with frame_lock:
            latest_frame = frame

        # Calculate Camera FPS
        cap_count += 1
        now = time.time()
        if now - cap_prev_time >= 1.0:
            with camera_lock:
                camera_fps = cap_count / (now - cap_prev_time)
            cap_count = 0
            cap_prev_time = now

    cap.release()
    print("[Capture] Stopped.")


def inference_thread_func():
    """
    Runs the AI model on the latest frame.
    Also handles logging and triggering the LED.
    """
    global latest_frame, latest_detections, stop_flag, infer_fps
    global last_detection_time, led_flashing, led

    # Initialize the database logger
    logger = DetectionLogger(DB_PATH)
    last_log_time = 0.0

    if not MODEL_PATH.exists():
        print(f"[Infer] Model not found: {MODEL_PATH.resolve()}")
        stop_flag = True
        return

    print("[Infer] Initializing Edge TPU interpreter...")
    # Load the TFLite model with the Edge TPU delegate (hardware acceleration)
    interpreter = tflite.Interpreter(
        model_path=str(MODEL_PATH),
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, ih, iw, _ = input_details[0]["shape"]
    print("[Infer] Interpreter ready.")

    infer_count = 0
    infer_prev_time = time.time()

    while not stop_flag:
        # Get the latest frame safely
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # Prepare image for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_q, scale, dw, dh = preprocess(frame_rgb, input_details)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], img_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Decode results
        detections = decode_yolo_output(
            output, output_details,
            orig_shape=frame.shape[:2],
            scale=scale,
            dw=dw,
            dh=dh,
            input_size=(ih, iw),
        )

        # Check for deer and trigger LED flashing
        # "deer" is class index 0 in COCO_NAMES
        deer_detected = any(cls == 0 for score, cls, *_ in detections)
        
        if deer_detected and led is not None:
            now = time.time()
            
            # Update LED state
            with led_lock:
                last_detection_time = now
                # Start blinking if not already flashing
                if not led_flashing:
                    led_flashing = True
                    # gpiozero blink is non-blocking (runs in its own thread)
                    led.blink(on_time=0.1, off_time=0.1)

            # --- Logging Logic ---
            # Only log if enough time has passed since the last log (cooldown)
            if now - last_log_time > LOG_COOLDOWN:
                last_log_time = now
                
                # 1. Save the image
                timestamp_str = datetime.now().astimezone().isoformat()
                filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                image_path = CAPTURE_DIR / filename
                cv2.imwrite(str(image_path), frame)
                
                # 2. Prepare data for the database
                det_list = []
                species_counts = {}
                
                for s, c, x1, y1, x2, y2 in detections:
                    c_idx = int(c)
                    species = COCO_NAMES[c_idx] if c_idx < len(COCO_NAMES) else f"class_{c_idx}"
                    
                    det_list.append({
                        "species": species,
                        "confidence": float(s),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    
                    species_counts[species] = species_counts.get(species, 0) + 1

                summary_parts = [f"{count} {spec}" for spec, count in species_counts.items()]
                summary_str = f"{len(detections)} animals detected: " + ", ".join(summary_parts)

                log_data = {
                    "timestamp": timestamp_str,
                    "device_id": DEVICE_ID,
                    "detected": True,
                    "night_mode": False,
                    "image_path": str(image_path),
                    "detections": det_list,
                    "species_count": species_counts,
                    "summary": summary_str
                }
                
                # 3. Write to database
                logger.log_detection(log_data)

        # Update shared detections list for the GUI thread
        with det_lock:
            latest_detections = detections

        # Calculate Inference FPS
        infer_count += 1
        now = time.time()
        if now - infer_prev_time >= 1.0:
            with infer_lock:
                infer_fps = infer_count / (now - infer_prev_time)
            infer_count = 0
            infer_prev_time = now

    print("[Infer] Stopped.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    """
```
    # Ensure LED is off when the program exits
    if led is not None:
        led.off()
    print("[LED] Manager thread stopped.")


# ------------------------------------------------------------
# Threads
# ------------------------------------------------------------
def capture_thread_func():
    """
    Continuously reads frames from the camera.
    This ensures we always have the freshest image ready for processing.
    """
    global latest_frame, stop_flag, camera_fps

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Capture] Could not open camera index {CAM_INDEX}")
        stop_flag = True
        return

    # Set resolution (optional, helps with speed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap_count = 0
    cap_prev_time = time.time()

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("[Capture] Failed to grab frame.")
            time.sleep(0.01)
            continue

        # Safely update the shared 'latest_frame' variable
        with frame_lock:
            latest_frame = frame

        # Calculate Camera FPS
        cap_count += 1
        now = time.time()
        if now - cap_prev_time >= 1.0:
            with camera_lock:
                camera_fps = cap_count / (now - cap_prev_time)
            cap_count = 0
            cap_prev_time = now

    cap.release()
    print("[Capture] Stopped.")


def inference_thread_func():
    """
    Runs the AI model on the latest frame.
    Also handles logging and triggering the LED.
    """
    global latest_frame, latest_detections, stop_flag, infer_fps
    global last_detection_time, led_flashing, led

    # Initialize the database logger
    logger = DetectionLogger(DB_PATH)
    last_log_time = 0.0

    if not MODEL_PATH.exists():
        print(f"[Infer] Model not found: {MODEL_PATH.resolve()}")
        stop_flag = True
        return

    print("[Infer] Initializing Edge TPU interpreter...")
    # Load the TFLite model with the Edge TPU delegate (hardware acceleration)
    interpreter = tflite.Interpreter(
        model_path=str(MODEL_PATH),
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, ih, iw, _ = input_details[0]["shape"]
    print("[Infer] Interpreter ready.")

    infer_count = 0
    infer_prev_time = time.time()

    while not stop_flag:
        # Get the latest frame safely
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # Prepare image for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_q, scale, dw, dh = preprocess(frame_rgb, input_details)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], img_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Decode results
        detections = decode_yolo_output(
            output, output_details,
            orig_shape=frame.shape[:2],
            scale=scale,
            dw=dw,
            dh=dh,
            input_size=(ih, iw),
        )

        # Check for deer and trigger LED flashing
        # "deer" is class index 0 in COCO_NAMES
        deer_detected = any(cls == 0 for score, cls, *_ in detections)
        
        if deer_detected and led is not None:
            now = time.time()
            
            # Update LED state
            with led_lock:
                last_detection_time = now
                # Start blinking if not already flashing
                if not led_flashing:
                    led_flashing = True
                    # gpiozero blink is non-blocking (runs in its own thread)
                    led.blink(on_time=0.1, off_time=0.1)

            # --- Logging Logic ---
            # Only log if enough time has passed since the last log (cooldown)
            if now - last_log_time > LOG_COOLDOWN:
                last_log_time = now
                
                # 1. Save the image
                timestamp_str = datetime.now().astimezone().isoformat()
                filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                image_path = CAPTURE_DIR / filename
                cv2.imwrite(str(image_path), frame)
                
                # 2. Prepare data for the database
                det_list = []
                species_counts = {}
                
                for s, c, x1, y1, x2, y2 in detections:
                    c_idx = int(c)
                    species = COCO_NAMES[c_idx] if c_idx < len(COCO_NAMES) else f"class_{c_idx}"
                    
                    det_list.append({
                        "species": species,
                        "confidence": float(s),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    
                    species_counts[species] = species_counts.get(species, 0) + 1

                summary_parts = [f"{count} {spec}" for spec, count in species_counts.items()]
                summary_str = f"{len(detections)} animals detected: " + ", ".join(summary_parts)

                log_data = {
                    "timestamp": timestamp_str,
                    "device_id": DEVICE_ID,
                    "detected": True,
                    "night_mode": False,
                    "image_path": str(image_path),
                    "detections": det_list,
                    "species_count": species_counts,
                    "summary": summary_str
                }
                
                # 3. Write to database
                logger.log_detection(log_data)

        # Update shared detections list for the GUI thread
        with det_lock:
            latest_detections = detections

        # Calculate Inference FPS
        infer_count += 1
        now = time.time()
        if now - infer_prev_time >= 1.0:
            with infer_lock:
                infer_fps = infer_count / (now - infer_prev_time)
            infer_count = 0
            infer_prev_time = now

    print("[Infer] Stopped.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    """
    Main entry point of the application.
    Initializes hardware, starts threads, and runs the GUI loop.
    """
    global stop_flag, latest_frame, latest_detections, infer_fps, camera_fps, led

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Object Detection on Raspberry Pi 5 with Coral Edge TPU")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    args = parser.parse_args()

    print(f"Using model: {MODEL_PATH}")
    stop_flag = False
    latest_frame = None
    latest_detections = []

    # Initialize LED hardware
    led = LED(LED_PIN)

    # Create threads
    cap_thread = threading.Thread(target=capture_thread_func)
    inf_thread = threading.Thread(target=inference_thread_func)
    led_thread = threading.Thread(target=led_manager_thread_func)

    # Start threads (not daemon threads for proper cleanup)
    cap_thread.start()
    inf_thread.start()
    led_thread.start()

    print("Press 'q' in the window to quit." if not args.headless else "Running in headless mode. Press Ctrl+C to quit.")

    prev_time = time.time()
    frame_count = 0
    disp_fps = 0.0

    try:
        # Main Loop
        while True:
            # If headless, we just sleep and let the other threads do the work
            if args.headless:
                time.sleep(1.0)
                continue

            # --- GUI Mode ---
            # Get the latest frame to display
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            # Get the latest detections to draw
            with det_lock:
                detections = list(latest_detections)

            # Draw boxes on the frame
            frame_drawn = draw_detections(frame, detections)

            # Display FPS calc (how fast we refresh the window)
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                disp_fps = frame_count / (curr_time - prev_time)
                frame_count = 0
                prev_time = curr_time

            # Read current inference FPS and camera FPS from other threads
            with infer_lock:
                inf_fps_local = infer_fps
            with camera_lock:
                cam_fps_local = camera_fps

            # Show all three
            text = f"Cam: {cam_fps_local:4.1f}  Inf: {inf_fps_local:4.1f}  Disp: {disp_fps:4.1f}"
            cv2.putText(
                frame_drawn, text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            # Show the window
            cv2.imshow("Coral YOLO Live (threaded)", frame_drawn)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received.")
    
    finally:
        # Signal threads to stop
        print("[Main] Shutting down...")
        stop_flag = True
        
        # Wait for threads to finish properly
        cap_thread.join(timeout=2.0)
        inf_thread.join(timeout=2.0)
        led_thread.join(timeout=2.0)
        
        # Clean up
        if not args.headless:
            cv2.destroyAllWindows()
        if led is not None:
            led.close()
        
        print("[Main] Exiting.")


if __name__ == "__main__":
    main()
```