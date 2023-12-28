import numpy as np
import cv2
from parameters import *  # Variables importing
import tensorflow as tf
from deepface import DeepFace
from collections import Counter
import time

image = cv2.imread('RB.jpg')

cv2.imshow('Deepface',image)

while True:
    cv2.imshow('Deepface', image)
    predictions = DeepFace.analyze(image)
    print(predictions)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
