"""
Enhanced Object Detection Module
Detects colored blocks with size classification (small vs large) and improved accuracy.
"""

import cv2
import numpy as np
import json
from collections import defaultdict
from google.genai import types


def put_text(img, text, org, scale=0.6, thickness=2):
    """Draw outlined text for better visibility."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)


def classify_block_size(area, width, height):
    """
    Classify block as 'small' or 'large' based on contour properties.

    Args:
        area: Contour area in pixels
        width: Bounding box width
        height: Bounding box height

    Returns:
        'small' or 'large'
    """
    # Calibrated thresholds - adjust based on your camera setup
    SMALL_MAX_AREA = 2000   # Small blocks typically < 2000 pixels
    LARGE_MIN_AREA = 3000   # Large blocks typically > 3000 pixels

    if area < SMALL_MAX_AREA:
        return "small"
    elif area > LARGE_MIN_AREA:
        return "large"
    else:
        # Medium zone - use aspect ratio and dimensions
        max_dim = max(width, height)
        if max_dim > 80:
            return "large"
        else:
            return "small"


def detect_blocks_with_size(frame, area_threshold=500):
    """
    Detect colored blocks and classify their sizes.

    Args:
        frame: Input BGR image
        area_threshold: Minimum area to consider as a valid block

    Returns:
        List of tuples: (label, (cx, cy), size, area, bbox)
        where label is like "small_blue1" or "large_red2"
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Enhanced color ranges for better detection
    color_ranges = {
        "blue": [(np.array([100, 120, 50]), np.array([130, 255, 255]))],
        "green": [(np.array([40, 70, 50]), np.array([80, 255, 255]))],
        "yellow": [(np.array([20, 120, 80]), np.array([35, 255, 255]))],
        "red": [
            (np.array([0, 120, 80]), np.array([10, 255, 255])),
            (np.array([160, 120, 80]), np.array([179, 255, 255]))
        ],
    }

    detected_blocks = []
    color_size_counts = defaultdict(int)  # Track counts like small_blue1, large_red2

    for color_name, ranges in color_ranges.items():
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_threshold:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Classify size
            size = classify_block_size(area, w, h)

            # Draw bounding box (different colors for different sizes)
            box_color = (0, 255, 0) if size == "small" else (255, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Calculate centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            # Create label with size prefix
            size_color_key = f"{size}_{color_name}"
            color_size_counts[size_color_key] += 1
            label_name = f"{size_color_key}{color_size_counts[size_color_key]}"

            # Annotate with size, color, and area
            put_text(frame, f"{label_name}", (x, y - 10))
            put_text(frame, f"area:{int(area)}", (x, y + h + 20))

            detected_blocks.append({
                "label": label_name,
                "center": (cx, cy),
                "size": size,
                "color": color_name,
                "area": area,
                "bbox": (x, y, w, h)
            })

    return detected_blocks


def capture_scene_with_enhanced_detection(
    cam_index=4,
    width: int = 640,
    height: int = 480,
    save_dir: str = "captures",
    capture_interval_sec: int = 10,
    area_threshold: int = 500,
):
    """
    Capture scene with enhanced detection including size classification.

    Returns JSON with detected blocks including size information.
    """
    try:
        import os
        import time

        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            return {"error": f"Could not open camera index {cam_index}"}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = "Enhanced Detection (q=quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        start_time = time.time()
        has_saved_once = False

        img_path = None
        json_path = None
        last_detected_blocks_data = []

        while True:
            ok, frame = cap.read()
            if not ok:
                ok, frame = cap.read()
                if not ok:
                    print("Frame grab failed.")
                    break

            detected_blocks = detect_blocks_with_size(frame, area_threshold)

            # Countdown/status text
            if has_saved_once:
                put_text(frame, "Capture done — labels only (q=quit)", (10, 60))
            else:
                elapsed = time.time() - start_time
                remaining = max(0, int(capture_interval_sec - elapsed))
                put_text(frame, f"Auto-save in {remaining}s (q=quit)", (10, 60))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # Auto-save when timer expires
            if (not has_saved_once) and (time.time() - start_time >= capture_interval_sec):
                has_saved_once = True

                img_path = os.path.join(save_dir, "capture_scene_enhanced.png")
                json_path = os.path.join(save_dir, "capture_scene_enhanced.json")

                cv2.imwrite(img_path, frame)
                print(f"Saved image: {img_path}")

                # Prepare JSON data
                data = []
                for block in detected_blocks:
                    cx, cy = block['center']
                    data.append({
                        "label": block['label'],
                        "x": cx,
                        "y": cy,
                        "size": block['size'],
                        "color": block['color'],
                        "area": int(block['area']),
                        "bbox": {
                            "x": block['bbox'][0],
                            "y": block['bbox'][1],
                            "w": block['bbox'][2],
                            "h": block['bbox'][3]
                        }
                    })

                last_detected_blocks_data = data

                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"Saved JSON: {json_path}")

        cap.release()
        cv2.destroyAllWindows()

        if img_path is None or json_path is None:
            return {
                "message": "Camera closed before auto-save completed.",
                "image_path": img_path,
                "json_path": json_path,
                "detected_blocks": last_detected_blocks_data,
            }

        return {
            "message": "Enhanced capture completed.",
            "image_path": img_path,
            "json_path": json_path,
            "detected_blocks": last_detected_blocks_data,
        }

    except Exception as e:
        return {"error": f"Error in capture_scene_with_enhanced_detection: {e}"}


schema_capture_scene_with_enhanced_detection = types.FunctionDeclaration(
    name="capture_scene_with_enhanced_detection",
    description=(
        "Opens the camera and detects colored blocks with size classification "
        "(small/large). Annotates with labels like 'small_blue1', 'large_red2', etc. "
        "Shows countdown, saves annotated frame and JSON with size information."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "cam_index": types.Schema(
                type=types.Type.INTEGER,
                description="Camera index (default 4).",
            ),
            "width": types.Schema(
                type=types.Type.INTEGER,
                description="Capture width in pixels (default 640).",
            ),
            "height": types.Schema(
                type=types.Type.INTEGER,
                description="Capture height in pixels (default 480).",
            ),
            "save_dir": types.Schema(
                type=types.Type.STRING,
                description="Directory for saving captures (default 'captures').",
            ),
            "capture_interval_sec": types.Schema(
                type=types.Type.INTEGER,
                description="Seconds before auto-save (default 10).",
            ),
            "area_threshold": types.Schema(
                type=types.Type.INTEGER,
                description="Minimum area threshold for block detection (default 500).",
            ),
        },
    ),
)


def parse_block_description(description: str, detections: list):
    """
    Parse natural language block descriptions and find matching blocks.

    Examples:
        "small blue block" -> finds first small_blue block
        "large yellow" -> finds first large_yellow block
        "the red block" -> finds first red block (any size)

    Args:
        description: Natural language description of block
        detections: List of detected blocks from JSON

    Returns:
        Matching block label or None
    """
    desc_lower = description.lower()

    # Extract size if specified
    size_pref = None
    if "small" in desc_lower:
        size_pref = "small"
    elif "large" in desc_lower or "big" in desc_lower:
        size_pref = "large"

    # Extract color
    color_pref = None
    colors = ["blue", "green", "yellow", "red"]
    for color in colors:
        if color in desc_lower:
            color_pref = color
            break

    # Find matching blocks
    matches = []
    for det in detections:
        label = det.get('label', '')
        det_size = det.get('size', '')
        det_color = det.get('color', '')

        size_match = (size_pref is None) or (size_pref == det_size)
        color_match = (color_pref is None) or (color_pref == det_color)

        if size_match and color_match:
            matches.append(det)

    if matches:
        return matches[0].get('label')
    return None


schema_parse_block_description = types.FunctionDeclaration(
    name="parse_block_description",
    description=(
        "Parses natural language block descriptions like 'small blue block' or "
        "'large yellow' and finds the matching block label from detections. "
        "Returns the label to use for pick-and-place operations."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "description": types.Schema(
                type=types.Type.STRING,
                description="Natural language description of the block (e.g., 'small blue block').",
            ),
            "detections": types.Schema(
                type=types.Type.ARRAY,
                description="List of detected blocks from enhanced detection JSON.",
                items=types.Schema(type=types.Type.OBJECT),
            ),
        },
    ),
)
