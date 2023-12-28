import numpy as np
import cv2
from parameters import *  # Variables importing
import tensorflow as tf
from deepface import DeepFace
from collections import Counter
import time

def model_info_display(frame, cascade_name, model_name):
    # Display model info
    cv2.rectangle(frame, (40, 30), (650, 80), (220, 220, 220), -1)
    cv2.putText(frame, "Face detection : " + cascade_name, (50, 50), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)
    cv2.putText(frame, "Model : " + model_name, (50, 50 + 20), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)


def most_frequent(list):
    counter = Counter(list)

    return counter.most_common(1)[0][0]

def insert_last(list, new_item):
    list = list[1:]
    list.append(new_item)
    return list

def face_img(img,face):
    (x,y,w,h) = face
    # res = img[y + 5:y + h - 5, x + 20:x + w - 20]   # WHY +5 and +20 ???
    res = img[y+0:y + h - 0, x + 0:x + w - 0]   # WHY +5 and +20 ???
    res = cv2.resize(res, (48, 48))

    return res

def face_detection(grey_frame, cascade, face_settings):
    faces = cascade.detectMultiScale(grey_frame, **face_settings)
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
    res = labels[predictions]
    return res

def emotion_detection_deepface(face_grey, model, labels):
    if face_grey.max() > 1:
        face_grey = face_grey / 255.0

    reshaped_face = face_grey.reshape(1, 48, 48, 1)
    predictions = model.predict(reshaped_face).argmax()
    res = labels[predictions]
    return res

def display_emotion(img, emotion, face, color, font, font_color):
    (x,y,w,h) = face
    cv2.rectangle(img, (x, y), (x + w // 3, y + 20), color, -1)
    cv2.putText(img, emotion, (x + 10, y + 15), font, 0.5, font_color, 2, cv2.LINE_AA)

def emotions_histogram(img,emotions_list, label, colors_emotions, font,origin, delta_y):
    # Create & Display Histogram
    counter = Counter(emotions_list)
    dict = {i: counter[i] for i in label}  # Sort counter values as the labels order in a dict

    (x0, y0) = origin
    i = 0
    for emot in dict:
        y = y0 - i * delta_y
        i += 1
        txt = emot + ' : ' + '*' * counter[emot]
        cv2.putText(img, txt, (x0, y), font, 0.8, colors_emotions[emot], 2, cv2.LINE_AA)


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

# # Set Window Size
cv2.namedWindow('Emotion Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Face detection settings
'data/Facial_expression_models/pre_trained_models/facial_expression_model_weights'
cascade1_ref = 'haar_cascade_face_detection.xml'
# face_model2 = 'haarcascade_frontalface_default.xml'
cascade2_ref = cascade1_ref
cascade1 = cv2.CascadeClassifier(cascade1_ref)
cascade2 = cv2.CascadeClassifier(cascade2_ref)

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 5,
    'minSize': (100, 100)
}

labels1 = ['Surprise', 'Neutral', 'Angry', 'Happy', 'Sad']
labels2 = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
labels3 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Histogram UI
bgr_colors = {
    'purple': (128, 0, 128),
    'blue': (255, 0, 0),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'grey': (128, 128, 128),
    'black': (0, 0, 0)
}
Colors_emotions = {
    'Surprise': bgr_colors['purple'],
    'Neutral':bgr_colors['grey'],
    'Angry':bgr_colors['red'],
    'Happy':bgr_colors['green'],
    'Sad':bgr_colors['blue'],
    'Disgust':bgr_colors['yellow'],
    'Fear':bgr_colors['orange']
}
histo_scale = 0.5

# Emotion Recognition models
Keras_model1 = 'network-5Labels.h5'
Keras_model2 = 'Emotion_little_vgg.h5'
model1 = tf.keras.models.load_model(Keras_model1)
model2 = tf.keras.models.load_model(Keras_model2)
model2 = DeepFace.build_model("Emotion")

toto = "data/Facial_expression_models/pre_trained_models/network-5Labels.h5"
model_keras_deep = tf.keras.models.load_model(toto)

# Emotions buffer
emotions1 = ['Neutral']*max_emo
emotions2 = emotions1
final_emotion1 = ""
final_emotion2 = ""

# Font Display Settings
font = cv2.FONT_HERSHEY_SIMPLEX
color = (245, 135, 66)
font_color = (255,255,255)


prev_frame_time = 0

while True:
    # Capture frame-by-frame and Flip it
    ret, frame = cam.read()  # Frame is a (HEIGHT, WIDTH, 3) array
    frame = cv2.flip(frame, 1)

    # Crop to fit
    cropped = frame[center[0] - (crop_dim[0]) // 2: center[0] + (crop_dim[0]) // 2,
              center[1] - (crop_dim[1]) // 2: center[1] + (crop_dim[1]) // 2]
    # print(cropped.shape)

    grey = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    # ------------ FACE DETECTION WITH KERAS ----------------
    img1 = cropped.copy()

    label = labels1
    model_name = 'Keras' + Keras_model1
    model = model_keras_deep
    cascade_ref = cascade1_ref
    cascade = cascade1
    img = img1


    # Display model info
    model_info_display(img, cascade_name=cascade_ref, model_name=model_name)

    # Face & Emotion detection
    face = face_detection(grey_frame=grey,cascade=cascade,face_settings=settings)
    if face.size != 0:
        face_grey=face_img(img=grey,face=face)
        emotion = emotion_detection(face_grey=face_grey, model=model, labels=label)
        draw_rect(frame=img,face=face,color=color,thickness=2)
        emotions1 = insert_last(list=emotions1, new_item=emotion)
        final_emotion1 = most_frequent(emotions1)

        display_emotion(img,emotion=final_emotion1,face=face,color=color,font=font,font_color=font_color)

        # Display grey face on bottom left
        face_display = cv2.cvtColor(face_grey, cv2.COLOR_GRAY2BGR)
        face_display = cv2.resize(face_display,(150,150))
        img[img.shape[0]-int(face_display.shape[0]):, 0:int(face_display.shape[1])] = face_display

    # Create & Display Histogram
    emotions_histogram(img=img, emotions_list=emotions1, label=label, colors_emotions=Colors_emotions, font=font,
                               origin=(20, img.shape[0] - 50), delta_y=50)

    # ------------ FACE DETECTION WITH DEEPFACE ----------------
    img2 = cropped.copy()

    label = labels3
    model_name = 'Deepface facial_expression_model_weights.h5'
    model = model2
    cascade_ref = cascade2_ref
    cascade = cascade2
    img = img2

    # Display model info
    model_info_display(img, cascade_name=cascade_ref, model_name=model_name)

    # Face & Emotion detection
    face = face_detection(grey_frame=grey,cascade=cascade,face_settings=settings)
    if face.size != 0:
        face_grey = face_img(img=grey, face=face)
        emotion = emotion_detection_deepface(face_grey=face_grey, model=model, labels=label)
        draw_rect(frame=img, face=face, color=color, thickness=2)
        emotions2 = insert_last(list=emotions2, new_item=emotion)
        final_emotion2 = most_frequent(emotions2)

        display_emotion(img, emotion=final_emotion2, face=face, color=color, font=font, font_color=font_color)

        # Display grey face on bottom left
        face_display = cv2.cvtColor(face_grey, cv2.COLOR_GRAY2BGR)
        face_display = cv2.resize(face_display, (150, 150))
        img[img.shape[0]-int(face_display.shape[0]):, 0:int(face_display.shape[1])] = face_display

    # Create & Display Histogram
    emotions_histogram(img=img, emotions_list=emotions2, label=label, colors_emotions=Colors_emotions, font=font,
                           origin=(20, img.shape[0] - 50), delta_y=50)


    # ------------ FINAL DISPLAY ----------------
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

    # FPS Display
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(Grid, str(fps), (500, 500), font, 1, (0,0,255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', Grid)
    # print("FPS : {}".format(fps))

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
