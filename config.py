import numpy as np

WORKING_DIR = "."

block_height = -57

default_port="/dev/tty.usbmodem4796317814332"

#  3x3 affine matrix for pixel -> robot (X,Y)
# M = np.array([
#      [6.00650232e-03 ,-4.84214952e-01,  3.80653329e+02],
#      [-4.69079919e-01  ,3.74996755e-03,  1.55349575e+02]
# ], dtype=np.float64)
M = np.array([
    [-7.03946756e-03, -4.69080162e-01, 4.0330111e+02],
    [-4.47296787e-01, 5.97834246e-03, 1.41960926e+02]
], dtype=np.float64)
                 
z_above = 50           # safe travel height (e.g. 100)
z_table = -57           # Z at table contact
block_height_mm = 10   # block physical thickness
block_length_mm = 18   # block physical length
stack_delta_mm = 8    # extra height when stacking (to avoid collision)
side_offset_mm = 10    # extra XY gap when placing beside

capture_wait_time = 1
camera_index = 0