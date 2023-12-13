import numpy as np
import cv2
from parameters import * # Variables importing

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

# Set Window Size
cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame-by-frame and Flip it
    ret, frame = cam.read()                    # Frame is a (HEIGHT, WIDTH, 3) array
    frame = cv2.flip(frame, 1)

    # Crop to fit
    cropped = frame[center[0] - (crop_dim[0]) // 2: center[0] + (crop_dim[0]) // 2, center[1] - (crop_dim[1]) // 2: center[1] + (crop_dim[1]) // 2]

    # frame = cv2.resize(frame,(960,1080)) #disize (WIDTH,HEIGHT) and not (HEIGHT, WIDTH) as img
    
    # Create Grey scale
    grey = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    grey_RGB = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    # Create two empty canvas
    background1 = np.zeros((window_height//2, window_width, 3), dtype=np.uint8)
    background2 = np.zeros((window_height//2, window_width, 3), dtype=np.uint8)

    # Fill the canvas
    center_bg = (background1.shape[0]/2, background1.shape[1]/2)
    background1[int(center_bg[0] - cropped.shape[0]/2): int(center_bg[0] + cropped.shape[0]/2), int(center_bg[1]-cropped.shape[1]/2):int(center_bg[1]+cropped.shape[1]/2)] = cropped
    background2[int(center_bg[0] - grey_RGB.shape[0]/2): int(center_bg[0] + grey_RGB.shape[0]/2), int(center_bg[1] - grey_RGB.shape[1] / 2):int(center_bg[1] + grey_RGB.shape[1] / 2)] = grey_RGB

    # The two background one on the other
    Grid = np.concatenate((background1, background2), axis=0)
    
    # Display the resulting frame
    cv2.imshow('Webcam', Grid)
    
    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
