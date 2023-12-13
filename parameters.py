"All parameters all here "
import pyautogui

# Variables
res_cam_height = 2160
res_cam_width = 3840

window_width, window_height = pyautogui.size()

crop_height = window_height//2
# crop_width = int(crop_height*9/16)
crop_width = 1080
crop_dim = (crop_height, crop_width)
center = (1000, res_cam_width//2)       # Crop Center // Adjust height


