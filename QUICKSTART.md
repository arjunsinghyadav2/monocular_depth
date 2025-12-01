# Quick Start Guide

Get your LLM-controlled robot running in 5 minutes!

## Prerequisites

- ✅ Dobot Magician robot connected via USB
- ✅ USB camera connected
- ✅ Python 3.8+ installed
- ✅ Google Gemini API key

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment

Create `.env` file:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Step 3: Verify Robot Connection

```bash
# Check available serial ports
ls /dev/ttyACM*

# Update config.py if needed
# default_port="/dev/ttyACM2"  # Change this to match your port
```

## Step 4: Test Camera

```bash
# List available cameras
ls /dev/video*

# Test camera at index 4
python -c "import cv2; cap = cv2.VideoCapture(4); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

## Step 5: Run the System

```bash
python LLM_ROBOT.py
```

## First Time Usage

When the system starts, it will:

1. **Initialize depth model** (downloads ~300MB on first run)
2. **Connect to robot**
3. **Move to home position**
4. **Capture scene** (10 second countdown)
5. **Display detected blocks**

Example output:
```
Detected blocks:
 - small_blue1 at (120, 250) depth: 0.65
 - large_red1 at (340, 280) depth: 0.72
 - small_yellow1 at (200, 150) depth: 0.58
```

## Try These Commands

### Example 1: Simple Pick and Place
```
You: Pick up the small blue block and place it on the large red block
```

The system will:
- Identify "small_blue1" and "large_red1"
- Execute pick-and-place operation
- Report completion

### Example 2: With Rotation
```
You: Pick up small blue block, rotate 90 degrees, and place on large red block
```

### Example 3: Relative Positioning
```
You: Place the small yellow block to the right of the red block
```

## Common Issues

### "Could not open camera"
- Check camera_index in config.py
- Try different indices: 0, 1, 2, 4, 6

### "Error connecting to Dobot"
- Check USB connection
- Verify port: `ls /dev/ttyACM*`
- Check permissions: `sudo chmod 666 /dev/ttyACM2`

### "Depth model initialization failed"
- Ensure internet connection (first run downloads model)
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Verify CUDA (optional): `python -c "import torch; print(torch.cuda.is_available())"`

## Next Steps

1. **Calibrate camera-robot transform**: See README.md calibration section
2. **Adjust color ranges**: Tune HSV values in Enhanced_Detection.py for your lighting
3. **Fine-tune block sizes**: Adjust area thresholds in Enhanced_Detection.py
4. **Train more prompts**: Try various command variations

## File Locations

After running, check:
- `captures/scene_complete.png` - Annotated scene
- `captures/scene_complete.json` - Detection data with depth
- `captures/scene_depth.png` - Depth visualization

## Getting Help

- Check README.md for detailed documentation
- Review system architecture and workflow
- Examine example outputs in captures/

## Tips

✨ **Best practices:**
- Use good lighting for better detection
- Place blocks on dark/contrasting background
- Keep workspace clear of obstacles
- Start with simple commands to test
- Use verbose mode for debugging: `python LLM_ROBOT.py --verbose`

🎯 **For best results:**
- Calibrate affine matrix M for your setup
- Adjust z_table and block_height_mm to match your workspace
- Fine-tune color ranges for your blocks
- Ensure camera has clear view of workspace

---

**Ready to start?** Run `python LLM_ROBOT.py` and say hello to your intelligent robot! 🤖
