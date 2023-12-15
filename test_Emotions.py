import numpy as np
import cv2
from parameters import *  # Variables importing
import tensorflow as tf
from deepface import DeepFace

def model_info_display(frame, face_name, model_name):
    # Display model info
    cv2.rectangle(frame, (40, 30), (650, 80), (220, 220, 220), -1)
    cv2.putText(frame, "Face detection : " + face_name, (50, 50), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)
    cv2.putText(frame, "Model : " + model_name, (50, 50 + 20), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)


def most_frequent(list):
    counter = 0
    num = list[0]

    for i in list:
        curr_frequency = list.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num

def insert_last(list, new_item):
    list = list[1:]
    list.append(new_item)
    return list

def face_img(img,face):
    (x,y,w,h) = face
    res = img[y + 5:y + h - 5, x + 20:x + w - 20]   # WHY +5 and +20 ???
    res = cv2.resize(res, (48, 48))

    return res

def face_detection(frame, face_detection, face_settings):
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_detection.detectMultiScale(grey, **face_settings)
    selected_face = np.array([])

    # Face selection
    if len(faces) != 0:
        selected_face = faces[0]
        for face in faces:
            if face[2]*face[3]>selected_face[2]*selected_face[3]:
                selected_face=face

    return selected_face

def draw_rect(frame, face, color,thickness):
    (x, y, w, h) = face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def emotion_detection(face_grey, model, labels):
    if face_grey.max() > 1:
        face_grey = face_grey / 255.0

    predictions = model.predict(np.array([face_grey.reshape((48, 48, 1))])).argmax()
    emotion = labels[predictions]
    return emotion


def emotion_detection_deepeface(face_grey, model, labels):
    if face_grey.max() > 1:
        face_grey = face_grey / 255.0

    reshaped_face = face_grey.reshape(1, 48, 48, 1)
    predictions = model.predict(reshaped_face).argmax()
    emotion = labels[predictions]
    return emotion

def display_emotion(img, emotion, face, color, font, font_color):
    (x,y,w,h) = face
    cv2.rectangle(img, (x, y), (x + w // 3, y + 20), color, -1)
    cv2.putText(img, emotion, (x + 10, y + 15), font, 0.5, font_color, 2, cv2.LINE_AA)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolutionw
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

# Set Window Size
cv2.namedWindow('Emotion Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Face detection settings
face_model1 = 'haar_cascade_face_detection.xml'
# face_model2 = 'haarcascade_frontalface_default.xml'
face_model2 = face_model1
face_detection1 = cv2.CascadeClassifier(face_model1)
face_detection2 = cv2.CascadeClassifier(face_model2)

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 5,
    'minSize': (100, 100)
}
labels1 = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
labels2 = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
labels3 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

Keras_model1 = 'network-5Labels.h5'
Keras_model2 = 'Emotion_little_vgg.h5'
model1 = tf.keras.models.load_model(Keras_model1)
model2 = tf.keras.models.load_model(Keras_model2)
model2 = DeepFace.build_model("Emotion")

emotions1 = ['Neutral']*max_emo
emotions2 = []

font = cv2.FONT_HERSHEY_SIMPLEX
color = (245, 135, 66)
font_color = (255,255,255)

while True:
    # Capture frame-by-frame and Flip it
    ret, frame = cam.read()  # Frame is a (HEIGHT, WIDTH, 3) array
    frame = cv2.flip(frame, 1)

    # Crop to fit
    cropped = frame[center[0] - (crop_dim[0]) // 2: center[0] + (crop_dim[0]) // 2,
              center[1] - (crop_dim[1]) // 2: center[1] + (crop_dim[1]) // 2]

    grey = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    # ---- FACE DETECTION WITH MODEL 1 ----
    img1 = cropped.copy()
    # Display model info
    model_info_display(img1, face_model1, model_name='Keras ' + Keras_model1)

    # Face & Emotion detection
    face = face_detection(frame=img1,face_detection=face_detection1,face_settings=settings)
    if face.size != 0:
        face_grey=face_img(img=grey,face=face)
        emotion = emotion_detection(face_grey=face_grey, model=model1, labels=labels1)
        draw_rect(frame=img1,face=face,color=color,thickness=2)
        emotions1 = insert_last(list=emotions1, new_item=emotion)
        final_emotion1 = most_frequent(emotions1)

        display_emotion(img1,emotion=final_emotion1,face=face,color=color,font=font,font_color=font_color)

        # Display grey face on bottom left
        face_display = cv2.cvtColor(face_grey, cv2.COLOR_GRAY2BGR)
        face_display = cv2.resize(face_display,(150,150))
        img1[img1.shape[0]-int(face_display.shape[0]):, 0:int(face_display.shape[1])] = face_display

    # ---- FACE DETECTION WITH DEEPFACE ----
    img2 = cropped.copy()
    # Display model info
    model_info_display(img2, face_model2, model_name='Deepface facial_expression_model_weights.h5')

    # Face & Emotion detection
    face = face_detection(frame=img2, face_detection=face_detection2, face_settings=settings)
    if face.size != 0:
        face_grey = face_img(img=grey, face=face)
        emotion = emotion_detection_deepeface(face_grey=face_grey, model=model2, labels=labels3)
        draw_rect(frame=img2, face=face, color=color, thickness=2)
        emotions2 = insert_last(list=emotions2, new_item=emotion)
        final_emotion2 = most_frequent(emotions2)

        display_emotion(img2, emotion=final_emotion2, face=face, color=color, font=font, font_color=font_color)

        # Display grey face on bottom left
        face_display = cv2.cvtColor(face_grey, cv2.COLOR_GRAY2BGR)
        face_display = cv2.resize(face_display, (150, 150))
        img2[img2.shape[0] - int(face_display.shape[0]):, 0:int(face_display.shape[1])] = face_display

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
    cv2.imshow('Emotion Detection', Grid)

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
