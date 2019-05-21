import cv2,time
import numpy

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml");
video = cv2.VideoCapture(0)




while True:


    check, frame = video.read()
    #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0), 2)

    cv2.imshow("Face", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()


cv2.destroyAllWindows()
