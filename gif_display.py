import cv2
from pathlib import Path
import random

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Chemin vers le GIF
gif_path = Path(r'assets\Reno_Emotions\angry.gif')

# Ouvrir le GIF avec OpenCV
gif = cv2.VideoCapture(str(gif_path))

gif_idx = 0
trigger_event = False


while True:
    ret,frame_cam = cam.read()

    if trigger_event:
        ret_gif, frame_gif = gif.read()  # Lecture de la frame courante
        if ret_gif:
            cv2.imshow('Webcam avec GIF', frame_gif)
            gif_idx += 1
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        else:
            # Revenir au début du GIF si la dernière frame est atteinte
            gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
            gif_idx = 0
            trigger_event = False
    else:
        cv2.imshow('Webcam', frame_cam)


    random_event = random.randint(0,1)
    print('random_event :',random_event)
    if random_event:
        trigger_event = True

    # Attendre 1 milliseconde entre chaque itération pour éviter de bloquer le système
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les captures vidéo et détruire toutes les fenêtres OpenCV
cam.release()
cv2.destroyAllWindows()