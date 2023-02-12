import numpy as np
import cv2 as cv
import os


harr = cv.CascadeClassifier('harr.xml')
p = []
for i in os.listdir('./photos'):
    p.append(i)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
IMAGE = '/home/bashardinho/Desktop/face_detection_project/photos/buble/download.jpeg'
img = cv.imread(IMAGE)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Person', gray)

# Detecting
face_rect = harr.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in face_rect:
    faces_roi = gray[y:y+h, x:x+h]

    lable, confidance = face_recognizer.predict(faces_roi)
    print(f"Lable = {p[lable]} ----- with {confidance} confidance")

    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv.putText(img, str(p[lable]), (20, 20),
               cv.FONT_ITALIC, 1.0, (255, 0, 0), thickness=2)


cv.imshow('Detected Person', img)
cv.waitKey(0)
