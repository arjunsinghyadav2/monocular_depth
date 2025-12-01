"""
Integrated Scene Understanding Module
Combines enhanced detection with depth estimation for comprehensive scene analysis.
"""

import cv2
import numpy as np
import torch
import json
import os
import time
from collections import defaultdict
from google.genai import types


def initialize_scene_understanding_system(model_type="DPT_Large"):
    """
    Initialize the complete scene understanding system including depth model.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MiDaS depth model
        depth_model = torch.hub.load("intel-isl/MiDaS", model_type)
        depth_model.to(device)
        depth_model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            depth_transform = midas_transforms.dpt_transform
        else:
            depth_transform = midas_transforms.small_transform

        # Store in global state
        global _depth_model, _depth_transform, _device
        _depth_model = depth_model
        _depth_transform = depth_transform
        _device = device

        return {
            "message": f"Scene understanding system initialized with {model_type} on {device}",
            "model": model_type,
            "device": str(device),
        }

    except Exception as e:
        return {"error": f"Failed to initialize scene understanding: {e}"}


schema_initialize_scene_understanding_system = types.FunctionDeclaration(
    name="initialize_scene_understanding_system",
    description=(
        "Initializes the complete scene understanding system including depth estimation. "
        "Must be called before using integrated scene analysis features."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "model_type": types.Schema(
                type=types.Type.STRING,
                description="Depth model: 'DPT_Large' (recommended), 'DPT_Hybrid', or 'MiDaS_small'.",
            ),
        },
    ),
)


# Global state for depth model
_depth_model = None
_depth_transform = None
_device = None


def put_text(img, text, org, scale=0.6, thickness=2):
    """Draw outlined text."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)


def classify_block_size(area, width, height):
    """Classify block as small or large."""
    SMALL_MAX_AREA = 2000
    LARGE_MIN_AREA = 2100

    if area < SMALL_MAX_AREA:
        return "small"
    elif area > LARGE_MIN_AREA:
        return "large"
    else:
        max_dim = max(width, height)
        return "large" if max_dim > 80 else "small"


def detect_blocks_with_size_and_depth(frame, depth_map=None, area_threshold=500):
    """
    Detect blocks with size classification and optional depth information.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
    color_size_counts = defaultdict(int)

    for color_name, ranges in color_ranges.items():
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_threshold:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            size = classify_block_size(area, w, h)

            box_color = (0, 255, 0) if size == "small" else (255, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            size_color_key = f"{size}_{color_name}"
            color_size_counts[size_color_key] += 1
            label_name = f"{size_color_key}{color_size_counts[size_color_key]}"

            # Get depth if available
            depth_value = None
            depth_normalized = None
            if depth_map is not None:
                region_size = 10
                y_safe = max(0, min(cy, depth_map.shape[0] - 1))
                x_safe = max(0, min(cx, depth_map.shape[1] - 1))

                y_start = max(0, y_safe - region_size)
                y_end = min(depth_map.shape[0], y_safe + region_size)
                x_start = max(0, x_safe - region_size)
                x_end = min(depth_map.shape[1], x_safe + region_size)

                depth_region = depth_map[y_start:y_end, x_start:x_end]
                depth_value = float(np.mean(depth_region))
                depth_normalized = float((depth_value - depth_map.min()) / (depth_map.max() - depth_map.min()))

                # Annotate with depth
                put_text(frame, f"{label_name} d:{depth_normalized:.2f}", (x, y - 10))
            else:
                put_text(frame, f"{label_name}", (x, y - 10))

            detected_blocks.append({
                "label": label_name,
                "center": (cx, cy),
                "size": size,
                "color": color_name,
                "area": area,
                "bbox": (x, y, w, h),
                "depth_raw": depth_value,
                "depth_normalized": depth_normalized,
            })

    return detected_blocks


def capture_and_analyze_complete_scene(
    cam_index=0,
    width: int = 640,
    height: int = 480,
    save_dir: str = "captures",
    capture_interval_sec: int = 10,
    area_threshold: int = 500,
    use_depth: bool = True,
):
    """
    Comprehensive scene capture with detection, size classification, and depth estimation.

    This is the main function for complete scene understanding.
    """
    global _depth_model, _depth_transform, _device

    try:
        os.makedirs(save_dir, exist_ok=True)

        if use_depth and _depth_model is None:
            return {
                "error": "Depth model not initialized. Call initialize_scene_understanding_system first."
            }

        cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            return {"error": f"Could not open camera index {cam_index}"}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = "Complete Scene Analysis (q=quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        start_time = time.time()
        has_saved_once = False

        img_path = None
        json_path = None
        depth_viz_path = None
        last_detected_blocks_data = []

        while True:
            ok, frame = cap.read()
            if not ok:
                ok, frame = cap.read()
                if not ok:
                    print("Frame grab failed.")
                    break

            display_frame = frame.copy()

            # Estimate depth if enabled
            depth_map = None
            if use_depth and _depth_model is not None:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_batch = _depth_transform(img_rgb).to(_device)

                    with torch.no_grad():
                        prediction = _depth_model(input_batch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img_rgb.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()

                    depth_map = prediction.cpu().numpy()
                except Exception as e:
                    print(f"Depth estimation failed: {e}")

            # Detect blocks
            detected_blocks = detect_blocks_with_size_and_depth(
                display_frame, depth_map, area_threshold
            )

            # Status text
            if has_saved_once:
                put_text(display_frame, "Capture done (q=quit)", (10, 60))
            else:
                elapsed = time.time() - start_time
                remaining = max(0, int(capture_interval_sec - elapsed))
                status = f"Auto-save in {remaining}s | Depth: {'ON' if use_depth else 'OFF'}"
                put_text(display_frame, status, (10, 60))

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # Auto-save
            if (not has_saved_once) and (time.time() - start_time >= capture_interval_sec):
                has_saved_once = True

                img_path = os.path.join(save_dir, "scene_complete.png")
                json_path = os.path.join(save_dir, "scene_complete.json")

                cv2.imwrite(img_path, display_frame)
                print(f"Saved image: {img_path}")

                # Save depth visualization
                if depth_map is not None:
                    depth_viz_path = os.path.join(save_dir, "scene_depth.png")
                    depth_normalized = cv2.normalize(
                        depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

                    # Overlay detections
                    for block in detected_blocks:
                        cx, cy = block['center']
                        label = block['label']
                        if block['depth_normalized'] is not None:
                            cv2.circle(depth_colored, (cx, cy), 8, (0, 255, 0), -1)
                            cv2.putText(
                                depth_colored, label, (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                            )

                    cv2.imwrite(depth_viz_path, depth_colored)
                    print(f"Saved depth visualization: {depth_viz_path}")

                # Prepare JSON
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
                        },
                        "depth_raw": block['depth_raw'],
                        "depth_normalized": block['depth_normalized'],
                    })

                last_detected_blocks_data = data

                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"Saved JSON: {json_path}")

        cap.release()
        cv2.destroyAllWindows()

        if img_path is None:
            return {"message": "Camera closed before capture"}

        return {
            "message": "Complete scene analysis finished",
            "image_path": img_path,
            "json_path": json_path,
            "depth_visualization": depth_viz_path,
            "detected_blocks": last_detected_blocks_data,
            "depth_enabled": use_depth,
        }

    except Exception as e:
        return {"error": f"Error in capture_and_analyze_complete_scene: {e}"}


schema_capture_and_analyze_complete_scene = types.FunctionDeclaration(
    name="capture_and_analyze_complete_scene",
    description=(
        "Performs complete scene analysis: captures image, detects blocks with size "
        "classification (small/large), estimates depth, and creates comprehensive JSON "
        "with all information. This is the recommended function for full scene understanding."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "cam_index": types.Schema(
                type=types.Type.INTEGER,
                description="Camera index (default 0).",
            ),
            "width": types.Schema(
                type=types.Type.INTEGER,
                description="Capture width (default 640).",
            ),
            "height": types.Schema(
                type=types.Type.INTEGER,
                description="Capture height (default 480).",
            ),
            "save_dir": types.Schema(
                type=types.Type.STRING,
                description="Save directory (default 'captures').",
            ),
            "capture_interval_sec": types.Schema(
                type=types.Type.INTEGER,
                description="Seconds before auto-save (default 10).",
            ),
            "area_threshold": types.Schema(
                type=types.Type.INTEGER,
                description="Minimum area for detection (default 500).",
            ),
            "use_depth": types.Schema(
                type=types.Type.BOOLEAN,
                description="Enable depth estimation (default True).",
            ),
        },
    ),
)
