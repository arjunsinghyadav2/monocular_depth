import numpy as np

WORKING_DIR = "."

block_height = -45

default_port="/dev/ttyACM2"

#  3x3 affine matrix for pixel -> robot (X,Y)
M = np.array([
     [6.00650232e-03 ,-4.84214952e-01,  3.80653329e+02],
     [-4.69079919e-01  ,3.74996755e-03,  1.55349575e+02]
], dtype=np.float64)
                 
z_above = 100           # safe travel height (e.g. 100)
z_table = -45           # Z at table contact
block_height_mm = 40   # block physical thickness
block_length_mm = 20   # block physical length
stack_delta_mm = 10    # extra height when stacking (to avoid collision)
side_offset_mm = 10    # extra XY gap when placing beside

capture_wait_time = 10
camera_index = 4