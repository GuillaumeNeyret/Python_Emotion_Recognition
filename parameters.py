"All parameters all here "
import pyautogui

# Variables
res_cam_height = 720   # 2160 for 4K Cam
res_cam_width = 1280    # 3840 for 4K Cam

window_width, window_height = pyautogui.size()

crop_height = window_height//2  # //2 Because we want to display 2 img on the same screen
crop_width = window_width
if crop_height>res_cam_height:
    crop_height=res_cam_height
crop_dim = (crop_height, crop_width)
center = (int(res_cam_height*0.5), res_cam_width//2)       # Crop Center // Adjust height


