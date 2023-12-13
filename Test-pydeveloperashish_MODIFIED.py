from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

def model_info_display(frame, face_name, model_name):
    # Display model info
    cv2.rectangle(frame, (40, 30), (650, 80), (220, 220, 220), -1)
    cv2.putText(frame, "Face detection : " + face_name, (50, 50), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)
    cv2.putText(frame, "Model : "+model_name , (50, 50 + 20), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA, False)

def detectanddisplay(frame,face_detection, face_settings, model, font, labels):
    img = frame.copy()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model =load_model('Emotion_little_vgg.h5')

labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 5,
    'minSize': (5, 5)
}
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # labels = []

    img1 = detectanddisplay(frame=frame,face_detection=face_classifier, face_settings=settings,model=model, labels=labels, font=font)
    #
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # faces = face_classifier.detectMultiScale(gray,1.3,5)

    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h,x:x+w]
    #     roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # # rect,face,image = face_detector(frame)
    #
    #
    #     if np.sum([roi_gray])!=0:
    #         roi = roi_gray.astype('float')/255.0
    #         roi = img_to_array(roi)
    #         roi = np.expand_dims(roi,axis=0)
    #
    #     # make a prediction on the ROI, then lookup the class
    #
    #         preds = classifier.predict(roi)[0]
    #         label=class_labels[preds.argmax()]
    #         label_position = (x,y)
    #         cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    #     else:
    #         cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























