import cv2
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from parameters import *  # Variables importing

def random_list(min,max, nb_items):
    res = [random.randint(min, max) for _ in range(nb_items)]
    return res

def int_to_emotion(lst, label):
    res = [label[i] for i in lst]
    return res

emotions_labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
bar_colors = ['purple','grey','red','green','blue']

toto = random_list(0,4,10)

emotions_values = int_to_emotion(toto,emotions_labels)

# Calcul des fréquences des émotions
counter = Counter(emotions_values)
emotions_counts = [counter[i] for i in emotions_labels] # Sort counter values as the labels order

# Création de l'histogramme
plt.figure(figsize=(10, 6))
plt.bar(emotions_labels, emotions_counts, color=bar_colors)
# plt.xlabel('Emotions')
# plt.ylabel('Nombre d\'occurrences')
# plt.title('Histogramme des émotions')

# plt.show()


# Rendre la figure de Matplotlib en une image
figure_canvas = FigureCanvas(plt.gcf())
figure_canvas.draw()

# Convertir l'histogramme en une image pour OpenCV
histogram_img = np.array(figure_canvas.renderer.buffer_rgba())
histogram_img = cv2.cvtColor(histogram_img,cv2.COLOR_RGBA2BGR)

scale = 0.7
# histogram_img = cv2.resize(histogram_img,(100,100))

histogram_img = cv2.resize(histogram_img,(int(histogram_img.shape[1]*scale),int(histogram_img.shape[0]*scale)))

# # Exemple d'utilisation dans une fenêtre OpenCV
# # Création d'une fenêtre OpenCV et affichage de l'image de l'histogramme
# cv2.namedWindow('Histogramme avec OpenCV', cv2.WINDOW_NORMAL)
# cv2.imshow('Histogramme avec OpenCV', histogram_img)
# print(histogram_img.shape)
# # Attendre la touche 'q' pour quitter
# cv2.waitKey(0)
# cv2.destroyAllWindows()




cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# Set Camera Resolutionw
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

# Set Window Size
cv2.namedWindow('Emotion Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    # Capture frame-by-frame and Flip it
    ret, frame = cam.read()  # Frame is a (HEIGHT, WIDTH, 3) array
    frame = cv2.flip(frame, 1)

    # Crop to fit
    cropped = frame[center[0] - (crop_dim[0]) // 2: center[0] + (crop_dim[0]) // 2,
              center[1] - (crop_dim[1]) // 2: center[1] + (crop_dim[1]) // 2]

    grey = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    img1 = cropped.copy()

    img2 = cropped.copy()

    # Create two empty canvas
    background1 = np.zeros((window_height // 2, window_width, 3), dtype=np.uint8)
    background2 = np.zeros((window_height // 2, window_width, 3), dtype=np.uint8)

    # Fill the canvas
    center_bg = (background1.shape[0] / 2, background1.shape[1] / 2)
    background1[int(center_bg[0] - img1.shape[0] / 2): int(center_bg[0] + img1.shape[0] / 2),
    int(center_bg[1] - img1.shape[1] / 2):int(center_bg[1] + img1.shape[1] / 2)] = img1
    background2[int(center_bg[0] - img2.shape[0] / 2): int(center_bg[0] + img2.shape[0] / 2),
    int(center_bg[1] - img2.shape[1] / 2):int(center_bg[1] + img2.shape[1] / 2)] = img2

    # Add the histogram
    background1[0:histogram_img.shape[0],0:histogram_img.shape[1]] = histogram_img

    # The two background one on the other
    Grid = np.concatenate((background1, background2), axis=0)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', Grid)

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
