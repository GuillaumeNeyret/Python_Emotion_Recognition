import numpy as np
import cv2
from parameters import *  # Variables importing
import tensorflow as tf

def model_info_display(frame, face_name, model_name):
    # Display model info
    cv2.rectangle(frame, (40, 30), (650, 80), (220, 220, 220), -1)
    cv2.putText(frame, "Face detection : " + face_name, (50, 50), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)
    cv2.putText(frame, "Model : "+model_name , (50, 50 + 20), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)

def detectanddisplay(frame,face_detection, face_settings, model, font, labels):
    img = frame.copy()
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    detected = face_detection.detectMultiScale(grey, **face_settings)

    for x, y, w, h in detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (245, 135, 66), 2)
        cv2.rectangle(img, (x, y), (x + w // 3, y + 20), (245, 135, 66), -1)
        face = grey[y + 5:y + h - 5, x + 20:x + w - 20]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0

        predictions = model.predict(np.array([face.reshape((48, 48, 1))])).argmax()
        state = labels[predictions]

        cv2.putText(img, state, (x + 10, y + 15), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return img

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

# Set Window Size
cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Face detection settings
face_model1 = 'haar_cascade_face_detection.xml'
face_model2 = 'haarcascade_frontalface_default.xml'
face_detection1 = cv2.CascadeClassifier(face_model1)
face_detection2 = cv2.CascadeClassifier(face_model2)

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 5,
    'minSize': (400, 400)
}
labels1 = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
labels2 = ['Angry','Happy','Neutral','Sad','Surprise']

Keras_model1 = 'network-5Labels.h5'
Keras_model2 = 'Emotion_little_vgg.h5'
model1 = tf.keras.models.load_model(Keras_model1)
model2 = tf.keras.models.load_model(Keras_model2)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame-by-frame and Flip it
    ret, frame = cam.read()  # Frame is a (HEIGHT, WIDTH, 3) array
    frame = cv2.flip(frame, 1)

    # Crop to fit
    cropped = frame[center[0] - (crop_dim[0]) // 2: center[0] + (crop_dim[0]) // 2,
              center[1] - (crop_dim[1]) // 2: center[1] + (crop_dim[1]) // 2]

    # ---- FACE DETECTION WITH MODEL 1 ----
    img1 = cropped.copy()
    # Display model info
    model_info_display(img1,face_model1,model_name='Keras '+Keras_model1)
    # Detect faces
    img1 = detectanddisplay(frame=img1,face_detection=face_detection1,face_settings=settings,model=model1,font=font, labels=labels1)

    # FACE DETECTION WITH MODEL 2
    img2 = cropped.copy()
    # Display model info
    model_info_display(img2, face_model1, model_name='Keras '+Keras_model2)
    # Detect faces
    img2 = detectanddisplay(frame=img2, face_detection=face_detection1, face_settings=settings, model=model2, font=font, labels=labels1)

    # Create two empty canvas
    background1 = np.zeros((window_height // 2, window_width, 3), dtype=np.uint8)
    background2 = np.zeros((window_height // 2, window_width, 3), dtype=np.uint8)

    # Fill the canvas
    center_bg = (background1.shape[0] / 2, background1.shape[1] / 2)
    background1[int(center_bg[0] - img1.shape[0] / 2): int(center_bg[0] + img1.shape[0] / 2),
    int(center_bg[1] - img1.shape[1] / 2):int(center_bg[1] + img1.shape[1] / 2)] = img1
    background2[int(center_bg[0] - img2.shape[0] / 2): int(center_bg[0] + img2.shape[0] / 2),
    int(center_bg[1] - img2.shape[1] / 2):int(center_bg[1] + img2.shape[1] / 2)] = img2

    # The two background one on the other
    Grid = np.concatenate((background1, background2), axis=0)

    # Display the resulting frame
    cv2.imshow('Webcam', Grid)

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
