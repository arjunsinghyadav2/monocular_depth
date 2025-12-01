"""
Monocular Depth Estimation Module
Uses MiDaS for depth estimation to make the robot kinematically aware.
"""

import cv2
import numpy as np
import torch
import json
from google.genai import types

# Global depth model
depth_model = None
depth_transform = None
device = None


def initialize_depth_model(model_type="DPT_Large"):
    """
    Initialize the MiDaS depth estimation model.

    Args:
        model_type: Model variant - "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
                   (DPT_Large is most accurate, MiDaS_small is fastest)
    """
    global depth_model, depth_transform, device

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MiDaS model
        depth_model = torch.hub.load("intel-isl/MiDaS", model_type)
        depth_model.to(device)
        depth_model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            depth_transform = midas_transforms.dpt_transform
        else:
            depth_transform = midas_transforms.small_transform

        return f"Depth model '{model_type}' initialized on {device}"

    except Exception as e:
        return f"Error initializing depth model: {e}"


schema_initialize_depth_model = types.FunctionDeclaration(
    name="initialize_depth_model",
    description=(
        "Initializes the MiDaS monocular depth estimation model. "
        "This must be called before using depth estimation features."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "model_type": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Model variant: 'DPT_Large' (most accurate), "
                    "'DPT_Hybrid' (balanced), or 'MiDaS_small' (fastest). "
                    "Default is 'DPT_Large'."
                ),
            ),
        },
    ),
)


def estimate_depth_from_image(image_path: str, output_path: str = None):
    """
    Estimate depth from an image file.

    Args:
        image_path: Path to input image
        output_path: Optional path to save depth visualization

    Returns:
        Dictionary with depth map info
    """
    global depth_model, depth_transform, device

    try:
        if depth_model is None:
            return {"error": "Depth model not initialized. Call initialize_depth_model first."}

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Could not read image: {image_path}"}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform and predict
        input_batch = depth_transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

        # Save visualization if requested
        if output_path:
            cv2.imwrite(output_path, depth_colored)

        return {
            "message": "Depth estimation completed",
            "image_shape": img.shape,
            "depth_shape": depth_map.shape,
            "depth_min": float(depth_map.min()),
            "depth_max": float(depth_map.max()),
            "depth_mean": float(depth_map.mean()),
            "visualization_saved": output_path if output_path else "not saved",
        }

    except Exception as e:
        return {"error": f"Error in depth estimation: {e}"}


schema_estimate_depth_from_image = types.FunctionDeclaration(
    name="estimate_depth_from_image",
    description=(
        "Estimates depth map from an image using monocular depth estimation. "
        "Returns depth statistics and optionally saves a visualization."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "image_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the input image file.",
            ),
            "output_path": types.Schema(
                type=types.Type.STRING,
                description="Optional path to save depth visualization (colored depth map).",
            ),
        },
    ),
)


def estimate_depth_with_detections(
    image_path: str,
    detection_json_path: str,
    output_json_path: str = "captures/capture_scene_with_depth.json",
    depth_viz_path: str = "captures/depth_visualization.png",
):
    """
    Estimate depth at detected object locations and augment detection JSON.

    Args:
        image_path: Path to captured scene image
        detection_json_path: Path to detection JSON with object positions
        output_json_path: Path to save augmented JSON with depth info
        depth_viz_path: Path to save depth visualization

    Returns:
        Dictionary with augmented detections including depth
    """
    global depth_model, depth_transform, device

    try:
        if depth_model is None:
            return {"error": "Depth model not initialized. Call initialize_depth_model first."}

        # Load detections
        with open(detection_json_path, 'r') as f:
            detections = json.load(f)

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Could not read image: {image_path}"}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Estimate depth
        input_batch = depth_transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Augment detections with depth
        augmented_detections = []
        for detection in detections:
            x = int(detection.get('x', 0))
            y = int(detection.get('y', 0))
            label = detection.get('label', 'unknown')

            # Get depth at object center (with bounds checking)
            y_safe = max(0, min(y, depth_map.shape[0] - 1))
            x_safe = max(0, min(x, depth_map.shape[1] - 1))

            # Average depth in a small region around the center
            region_size = 10
            y_start = max(0, y_safe - region_size)
            y_end = min(depth_map.shape[0], y_safe + region_size)
            x_start = max(0, x_safe - region_size)
            x_end = min(depth_map.shape[1], x_safe + region_size)

            depth_region = depth_map[y_start:y_end, x_start:x_end]
            depth_value = float(np.mean(depth_region))

            augmented_detection = {
                "label": label,
                "x": x,
                "y": y,
                "depth_raw": depth_value,
                "depth_normalized": float((depth_value - depth_map.min()) / (depth_map.max() - depth_map.min())),
            }
            augmented_detections.append(augmented_detection)

        # Save augmented detections
        with open(output_json_path, 'w') as f:
            json.dump(augmented_detections, f, indent=2)

        # Create visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

        # Overlay detections on depth map
        for det in augmented_detections:
            x, y = det['x'], det['y']
            label = det['label']
            depth_norm = det['depth_normalized']

            cv2.circle(depth_colored, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                depth_colored,
                f"{label} d:{depth_norm:.2f}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(depth_viz_path, depth_colored)

        return {
            "message": "Depth estimation with detections completed",
            "detections_count": len(augmented_detections),
            "output_json": output_json_path,
            "depth_visualization": depth_viz_path,
            "detections": augmented_detections,
        }

    except Exception as e:
        return {"error": f"Error in depth estimation with detections: {e}"}


schema_estimate_depth_with_detections = types.FunctionDeclaration(
    name="estimate_depth_with_detections",
    description=(
        "Estimates depth at each detected object location and creates an augmented "
        "detection JSON with depth information. This makes the robot kinematically "
        "aware of object positions in 3D space."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "image_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the captured scene image.",
            ),
            "detection_json_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the detection JSON with object positions.",
            ),
            "output_json_path": types.Schema(
                type=types.Type.STRING,
                description="Path to save augmented JSON with depth info (default: captures/capture_scene_with_depth.json).",
            ),
            "depth_viz_path": types.Schema(
                type=types.Type.STRING,
                description="Path to save depth visualization (default: captures/depth_visualization.png).",
            ),
        },
    ),
)


def calculate_relative_depth_offset(
    detection_json_path: str,
    robot_current_height: float,
    target_label: str,
):
    """
    Calculate the Z-axis offset needed to reach a target block based on relative depth.

    This function helps the robot understand if it needs to move up or down
    to reach the target block from its current position.

    Args:
        detection_json_path: Path to depth-augmented detection JSON
        robot_current_height: Current Z height of robot end-effector
        target_label: Label of the target block

    Returns:
        Dictionary with depth offset information
    """
    try:
        with open(detection_json_path, 'r') as f:
            detections = json.load(f)

        # Find target block
        target_block = None
        for det in detections:
            if det.get('label') == target_label:
                target_block = det
                break

        if target_block is None:
            return {
                "error": f"Target block '{target_label}' not found",
                "available_labels": [d.get('label') for d in detections],
            }

        target_depth = target_block.get('depth_normalized', 0.5)

        # Find reference depths (min/max in scene)
        all_depths = [d.get('depth_normalized', 0.5) for d in detections]
        min_depth = min(all_depths)
        max_depth = max(all_depths)

        # Calculate relative position
        # depth_normalized: 0 = far, 1 = close
        # We can estimate if target is higher or lower than current position

        # This is a simplified heuristic - in practice, you'd calibrate this
        # For now, we assume: higher depth_normalized = closer to camera = higher Z
        relative_depth_score = (target_depth - min_depth) / (max_depth - min_depth + 1e-6)

        return {
            "target_label": target_label,
            "target_depth_normalized": target_depth,
            "relative_depth_score": relative_depth_score,
            "depth_range": {"min": min_depth, "max": max_depth},
            "all_detections": detections,
            "message": (
                f"Target '{target_label}' has depth score {relative_depth_score:.2f} "
                f"(0=farthest, 1=closest in scene)"
            ),
        }

    except Exception as e:
        return {"error": f"Error calculating depth offset: {e}"}


schema_calculate_relative_depth_offset = types.FunctionDeclaration(
    name="calculate_relative_depth_offset",
    description=(
        "Calculates relative depth information for a target block to help the robot "
        "understand vertical positioning. Returns depth score and relative positioning."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "detection_json_path": types.Schema(
                type=types.Type.STRING,
                description="Path to depth-augmented detection JSON.",
            ),
            "robot_current_height": types.Schema(
                type=types.Type.NUMBER,
                description="Current Z height of robot end-effector (mm).",
            ),
            "target_label": types.Schema(
                type=types.Type.STRING,
                description="Label of the target block (e.g., 'blue1').",
            ),
        },
    ),
)
