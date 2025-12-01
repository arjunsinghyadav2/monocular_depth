"""
Enhanced Pick and Place Tool with Rotation Support
Supports pick-and-place operations with rotation capabilities.
"""

import os
import json
from google.genai import types
from config import (
    M,                 # 3x3 affine matrix for pixel -> robot (X,Y)
    z_above,           # safe travel height (e.g. 100)
    z_table,           # Z at table contact
    block_height_mm,   # block physical thickness
    block_length_mm,   # block physical length
    stack_delta_mm,    # extra height when stacking (to avoid collision)
    side_offset_mm,    # extra XY gap when placing beside
)

from Robot_Tools.Robot_Motion_Tools import (
    move_robot_point_above,
    move_robot_point_block,
    suction_on,
    suction_off,
    apply_affine,
    move_to_specific_position,
    _ensure_device,
)

from pydobot.dobot import MODE_PTP
import time


def pick_and_place_with_rotation(
    detection_json_path: str = "captures/capture_scene_enhanced.json",
    source_label: str = "small_blue1",
    target_label: str = "large_red1",
    placement_type: str = "on_top",   # "on_top" or "beside"
    direction: str = "right",         # used for "beside": "front", "back", "right", "left"
    rotation_degrees: float = 0.0,    # rotation in Z-axis (degrees)
):
    """
    Enhanced pick-and-place with rotation support.

    Args:
        detection_json_path: Path to detection JSON
        source_label: Label of block to pick (e.g., 'small_blue1')
        target_label: Label of reference block (e.g., 'large_red1')
        placement_type: 'on_top' or 'beside'
        direction: For 'beside': 'front', 'back', 'right', 'left'
        rotation_degrees: Rotation angle in degrees (around Z-axis)

    Returns:
        Dictionary with operation results
    """
    try:
        if not os.path.exists(detection_json_path):
            return {
                "error": f"detection_json_path not found: {detection_json_path}"
            }

        with open(detection_json_path, "r") as f:
            detections = json.load(f)

        # Build lookup: label -> (u, v)
        label_to_uv = {}
        for item in detections:
            label = item.get("label")
            x = item.get("x")
            y = item.get("y")
            if label is not None and x is not None and y is not None:
                label_to_uv[label] = (x, y)

        if source_label not in label_to_uv:
            return {
                "error": f"{source_label} not found in detection file",
                "available_labels": list(label_to_uv.keys()),
            }

        if target_label not in label_to_uv:
            return {
                "error": f"{target_label} not found in detection file",
                "available_labels": list(label_to_uv.keys()),
            }

        src_u, src_v = label_to_uv[source_label]
        tgt_u, tgt_v = label_to_uv[target_label]

        # Get size info if available
        source_info = next((d for d in detections if d.get('label') == source_label), {})
        target_info = next((d for d in detections if d.get('label') == target_label), {})

        source_size = source_info.get('size', 'unknown')
        target_size = target_info.get('size', 'unknown')

        # -----------------------------------------
        # HEIGHT CALCULATION (depth-aware)
        # -----------------------------------------
        # Pick from top of block sitting on the table
        pickup_height = z_table + block_height_mm

        # Adjust pickup height based on depth if available
        source_depth = source_info.get('depth_normalized', None)
        if source_depth is not None:
            # Small adjustment based on depth (calibration may be needed)
            depth_adjustment = 0  # Can be enhanced with calibration
            pickup_height += depth_adjustment

        if placement_type == "on_top":
            # Place on top of target block
            if target_size == "large":
                # Larger blocks might be taller - adjust if needed
                place_height = z_table + 2 * block_height_mm + stack_delta_mm
            else:
                place_height = z_table + 2 * block_height_mm + stack_delta_mm

        elif placement_type == "beside":
            place_height = z_table + block_height_mm
        else:
            return {
                "error": f"Unknown placement_type '{placement_type}'. Use 'on_top' or 'beside'."
            }

        steps = []

        # -----------------------------------------
        # PICK SEQUENCE
        # -----------------------------------------
        steps.append({"step": "move_above_source", "u": src_u, "v": src_v})
        steps.append(move_robot_point_above(src_u, src_v, z_above))

        steps.append({"step": "descend_to_pick", "z": pickup_height})
        steps.append(move_robot_point_block(src_u, src_v, pickup_height))

        steps.append({"step": "suction_on"})
        steps.append(suction_on())

        steps.append({"step": "lift_after_pick", "z": z_above})
        steps.append(move_robot_point_above(src_u, src_v, z_above))

        # -----------------------------------------
        # ROTATION (if specified)
        # -----------------------------------------
        if abs(rotation_degrees) > 0.1:
            steps.append({
                "step": "rotate_block",
                "rotation": rotation_degrees,
                "description": f"Rotating block by {rotation_degrees} degrees in Z-axis"
            })

            # Get current position and rotate
            dev = _ensure_device()
            pose, _ = dev.get_pose()
            current_x, current_y, current_z, current_r = pose

            # Apply rotation
            new_r = current_r + rotation_degrees

            steps.append(
                move_to_specific_position(
                    x=current_x,
                    y=current_y,
                    z=current_z,
                    r=new_r
                )
            )

        # -----------------------------------------
        # PLACE SEQUENCE
        # -----------------------------------------
        if placement_type == "on_top":
            steps.append({"step": "move_above_target_on_top", "u": tgt_u, "v": tgt_v})

            # Move above target with rotation maintained
            if abs(rotation_degrees) > 0.1:
                # Use specific position to maintain rotation
                tgt_X, tgt_Y = apply_affine(M, tgt_u, tgt_v)
                steps.append(
                    move_to_specific_position(
                        x=tgt_X,
                        y=tgt_Y,
                        z=z_above,
                        r=rotation_degrees
                    )
                )
            else:
                steps.append(move_robot_point_above(tgt_u, tgt_v, z_above))

            steps.append({"step": "descend_to_place_on_top", "z": place_height})

            if abs(rotation_degrees) > 0.1:
                tgt_X, tgt_Y = apply_affine(M, tgt_u, tgt_v)
                steps.append(
                    move_to_specific_position(
                        x=tgt_X,
                        y=tgt_Y,
                        z=place_height,
                        r=rotation_degrees
                    )
                )
            else:
                steps.append(move_robot_point_block(tgt_u, tgt_v, place_height))

            place_info = {
                "mode": "on_top",
                "target_pixel": {"u": tgt_u, "v": tgt_v},
                "rotation_applied": rotation_degrees,
            }

        else:  # placement_type == "beside"
            tgt_X, tgt_Y = apply_affine(M, tgt_u, tgt_v)

            offset = block_length_mm + side_offset_mm

            dX = 0.0
            dY = 0.0
            dir_norm = (direction or "").lower()

            if dir_norm == "front":
                dX = offset
            elif dir_norm == "back":
                dX = -offset
            elif dir_norm == "right":
                dY = offset
            elif dir_norm == "left":
                dY = -offset
            else:
                return {
                    "error": f"Unknown direction '{direction}'. Use 'front', 'back', 'right', or 'left'."
                }

            place_X = tgt_X + dX
            place_Y = tgt_Y + dY

            steps.append({
                "step": "move_above_target_beside",
                "X": place_X,
                "Y": place_Y,
                "Z": z_above,
            })
            steps.append(
                move_to_specific_position(
                    x=place_X,
                    y=place_Y,
                    z=z_above,
                    r=rotation_degrees,
                )
            )

            steps.append({
                "step": "descend_to_place_beside",
                "X": place_X,
                "Y": place_Y,
                "Z": place_height,
            })
            steps.append(
                move_to_specific_position(
                    x=place_X,
                    y=place_Y,
                    z=place_height,
                    r=rotation_degrees,
                )
            )

            place_info = {
                "mode": "beside",
                "direction": dir_norm,
                "target_pixel": {"u": tgt_u, "v": tgt_v},
                "place_robot": {"x": place_X, "y": place_Y, "z": place_height},
                "rotation_applied": rotation_degrees,
            }

        # Common: release and lift
        steps.append({"step": "suction_off"})
        steps.append(suction_off())

        steps.append({"step": "lift_after_place", "z": z_above})
        if placement_type == "on_top":
            if abs(rotation_degrees) > 0.1:
                tgt_X, tgt_Y = apply_affine(M, tgt_u, tgt_v)
                steps.append(
                    move_to_specific_position(
                        x=tgt_X,
                        y=tgt_Y,
                        z=z_above,
                        r=rotation_degrees
                    )
                )
            else:
                steps.append(move_robot_point_above(tgt_u, tgt_v, z_above))
        else:
            steps.append(
                move_to_specific_position(
                    x=place_info["place_robot"]["x"],
                    y=place_info["place_robot"]["y"],
                    z=z_above,
                    r=rotation_degrees,
                )
            )

        return {
            "message": f"Placed {source_label} ({source_size}) on/near {target_label} ({target_size}) with {rotation_degrees}° rotation",
            "source_label": source_label,
            "source_size": source_size,
            "target_label": target_label,
            "target_size": target_size,
            "source_pixel": {"u": src_u, "v": src_v},
            "rotation_degrees": rotation_degrees,
            "heights": {
                "pickup_height": pickup_height,
                "place_height": place_height,
                "z_above": z_above,
            },
            "config_constants": {
                "z_table": z_table,
                "block_height_mm": block_height_mm,
                "block_length_mm": block_length_mm,
                "stack_delta_mm": stack_delta_mm,
                "side_offset_mm": side_offset_mm,
            },
            "placement": place_info,
            "steps": steps,
        }

    except Exception as e:
        return {"error": f"pick_and_place_with_rotation failed: {e}"}


schema_pick_and_place_with_rotation = types.FunctionDeclaration(
    name="pick_and_place_with_rotation",
    description=(
        "Pick a block and place it on top of or beside another block with optional "
        "rotation. Supports size-aware operations (small/large blocks) and Z-axis rotation. "
        "Use this for commands like 'pick up the small blue block, rotate by 90 degrees, "
        "and place it on the large red block'."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "detection_json_path": types.Schema(
                type=types.Type.STRING,
                description="Path to enhanced detection JSON (default 'captures/capture_scene_enhanced.json').",
            ),
            "source_label": types.Schema(
                type=types.Type.STRING,
                description="Label of block to pick (e.g., 'small_blue1').",
            ),
            "target_label": types.Schema(
                type=types.Type.STRING,
                description="Label of reference block (e.g., 'large_red1').",
            ),
            "placement_type": types.Schema(
                type=types.Type.STRING,
                description="'on_top' to stack or 'beside' to place next to target.",
            ),
            "direction": types.Schema(
                type=types.Type.STRING,
                description="For 'beside' placement: 'front', 'back', 'right', or 'left'.",
            ),
            "rotation_degrees": types.Schema(
                type=types.Type.NUMBER,
                description="Rotation angle in degrees around Z-axis (e.g., 90 for 90° rotation).",
            ),
        },
    ),
)
