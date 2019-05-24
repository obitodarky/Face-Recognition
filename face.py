import time
import numpy
import  cv2
import pickle


face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml");
video = cv2.VideoCapture(0)

labels = {}
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

while True:


    check, frame = video.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5)


    for x,y,w,h in faces:
        id_, conf = recognizer.predict(gray_image[y: y + h, x: x + w])
        if conf >= 70 and conf <= 85:
            #print(id_)
            print(labels[id_])
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0), 2)

    cv2.imshow("Face", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()


cv2.destroyAllWindows()
