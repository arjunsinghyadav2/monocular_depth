# LLM-Controlled Robot with Monocular Depth Estimation

An advanced robotic control system that uses Large Language Models (LLMs) and Vision Language Models (VLMs) with monocular depth estimation for kinematically-aware pick-and-place operations.

## Features

- **Natural Language Control**: Control the robot using natural language commands
- **Monocular Depth Estimation**: Uses MiDaS DPT-Large for 3D scene understanding
- **Size-Aware Object Detection**: Detects and classifies blocks as small or large
- **Color Detection**: Supports blue, green, yellow, and red blocks
- **Rotation Support**: Can rotate objects during pick-and-place (90°, 180°, 270°)
- **Kinematic Awareness**: Uses depth information for height-aware positioning
- **Agentic AI Framework**: Powered by Google Gemini 2.5 Flash with function calling

## Requirements

- Python 3.8+
- Dobot Magician robotic arm
- USB camera (configured at index 4 by default)
- CUDA-capable GPU (recommended for faster depth estimation)
- Google Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd monocular_depth
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Gemini API key:
```bash
GEMINI_API_KEY=api_key_here
```

4. Configure robot settings in `config.py`:
   - Set correct serial port (default: `/dev/ttyACM2`)
   - Adjust camera index if needed
   - Calibrate affine transformation matrix `M` for your setup

## Usage

### Basic Usage

Run the interactive robot control system:
```bash
python LLM_ROBOT.py
```

### With Initial Command

```bash
python LLM_ROBOT.py "Pick up the small blue block and place it on the large red block"
```

### Verbose Mode

```bash
python LLM_ROBOT.py "move home" --verbose
```

## Example Commands

The system supports various natural language commands:

1. **Simple Pick and Place**:
   ```
   Pick up the small blue block and place it in the box on the right.
   ```

2. **Size-Specific Commands**:
   ```
   Pick up the large yellow block and place it in the box on the left.
   ```

3. **Stacking**:
   ```
   Pick up a small block and place it on top of the large block.
   ```

4. **With Rotation**:
   ```
   Pick up the small blue block, rotate by 90 degrees in z, and place it on large red block.
   ```

5. **Relative Positioning**:
   ```
   Pick a small yellow block and place it to the right of the red block.
   ```

## System Architecture

### Core Modules

1. **LLM_ROBOT.py**: Main interactive loop with Gemini AI
2. **call_function.py**: Function calling dispatcher
3. **config.py**: Configuration and calibration parameters

### Robot Tools

- **Robot_Motion_Tools.py**: Basic robot control (move, home, suction)
- **Camera_Capture_Tools.py**: Basic camera capture and detection
- **Enhanced_Detection.py**: Size-aware block detection
- **Depth_Estimation.py**: Monocular depth estimation using MiDaS
- **Pick_Place_With_Rotation.py**: Advanced pick-and-place with rotation
- **Integrated_Scene_Understanding.py**: Combined detection + depth analysis

### Helper Functions

- **file_handling.py**: File operations for the LLM

## Workflow

### Standard Pick-and-Place Workflow

1. **Initialization** (first time only):
   - Initialize scene understanding system (loads MiDaS model)
   - Connect to robot
   - Move to home position

2. **Scene Capture**:
   - `capture_and_analyze_complete_scene()` captures image
   - Detects blocks with size classification
   - Estimates depth for each object
   - Saves `scene_complete.json` and `scene_depth.png`

3. **Command Parsing**:
   - LLM parses natural language command
   - Identifies source and target blocks
   - Determines action type (on_top/beside)
   - Extracts rotation if specified

4. **Execution**:
   - `pick_and_place_with_rotation()` executes the operation
   - Uses depth-aware positioning
   - Applies rotation if specified
   - Reports completion

## Depth-Aware Positioning

The system uses monocular depth estimation to:
- Understand relative positions of objects in 3D
- Verify reachability before attempting pick
- Adjust approach heights based on scene geometry
- Provide kinematic awareness for safe operation

**Depth Values**:
- `depth_normalized`: 0.0 (far) to 1.0 (close to camera)
- Used to infer relative heights and positions
- Helps avoid collisions and unreachable targets

## Object Detection

### Color Ranges (HSV)
- **Blue**: [100, 120, 50] - [130, 255, 255]
- **Green**: [40, 70, 50] - [80, 255, 255]
- **Yellow**: [20, 120, 80] - [35, 255, 255]
- **Red**: [0, 120, 80] - [10, 255, 255] and [160, 120, 80] - [179, 255, 255]

### Size Classification
- **Small blocks**: Area < 2000 pixels²
- **Large blocks**: Area > 3000 pixels²
- Intermediate sizes classified by bounding box dimensions

## Coordinate Transformations

The system uses an affine transformation matrix `M` to convert:
- **Pixel coordinates (u, v)** → **Robot coordinates (X, Y)**

This matrix must be calibrated for your specific camera-robot setup.

## Configuration Parameters

In `config.py`:
```python
z_above = 100           # Safe travel height (mm)
z_table = -45           # Table surface height (mm)
block_height_mm = 40    # Block thickness (mm)
block_length_mm = 20    # Block length (mm)
stack_delta_mm = 10     # Extra clearance when stacking (mm)
side_offset_mm = 10     # Gap when placing beside (mm)
```

## Calibration

### Camera-Robot Calibration

1. Place calibration markers at known positions
2. Capture image and note pixel coordinates
3. Measure actual robot coordinates
4. Compute affine transformation matrix `M`
5. Update `config.py`

### Depth Calibration

The depth estimation is relative and normalized:
- No absolute depth measurements
- Comparative depth between objects
- Can be enhanced with ground truth measurements

## Output Files

Generated in `captures/` directory:
- `scene_complete.png`: Annotated scene image
- `scene_complete.json`: Detection data with depth
- `scene_depth.png`: Depth visualization with overlays

## Testing

Test individual components:
```bash
# Test robot connection
python -c "from Robot_Tools.Robot_Motion_Tools import *; print(get_dobot_device())"

# Test depth model
python -c "from Robot_Tools.Depth_Estimation import *; print(initialize_depth_model())"

# Test camera
python -c "from Robot_Tools.Camera_Capture_Tools import *; capture_scene_with_detection()"
```

## Troubleshooting

### Robot Not Connecting
- Check USB port in `config.py` (default: `/dev/ttyACM2`)
- Verify robot is powered on
- Check permissions: `sudo chmod 666 /dev/ttyACM2`

### Camera Not Found
- Adjust `camera_index` in `config.py`
- List cameras: `ls /dev/video*`
- Test camera: `python -c "import cv2; print(cv2.VideoCapture(4).isOpened())"`

### Depth Model Errors
- Ensure PyTorch is installed correctly
- Check CUDA availability for GPU acceleration
- First run downloads MiDaS models (~300MB)

### Detection Issues
- Adjust HSV color ranges for your lighting
- Modify `area_threshold` for your block sizes
- Ensure good lighting and contrast with background

## Contributing

Contributions are welcome! Areas for improvement:
- Multi-object manipulation sequences
- Collision avoidance
- Better depth calibration methods
- Support for more object shapes
- Voice command integration

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **MiDaS**: Intel ISL for monocular depth estimation
- **Google Gemini**: For powerful LLM function calling
- **PyDobot**: For Dobot robot control library
- **OpenCV**: For computer vision capabilities

## Contact

For issues and questions, please open a GitHub issue.
---

**Built with love for intelligent robotics**