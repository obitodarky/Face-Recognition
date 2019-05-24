import os
import cv2
import numpy as np
import pickle
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #path for current directory

image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")

recognize_images = cv2.face.LBPHFaceRecognizer_create()

labels_list = []
train_list = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root)

            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            #print(label_ids)
            pil_image = Image.open(path).convert("L")
            size = (200,200)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, 'uint8')

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                train_list.append(roi)
                labels_list.append(id_)


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)


recognize_images.train(train_list, np.array(labels_list))
recognize_images.save("trainner.yml")
