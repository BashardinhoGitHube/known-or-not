import os
import numpy as np
import cv2 as cv


p = []
DIR = r'/home/bashardinho/Desktop/face_detection_project/photos'
for i in os.listdir(r'./photos'):
    p.append(i)
    # print(i)
harr = cv.CascadeClassifier('harr.xml')
features = []
lables = []


def create_train():
    for person in p:
        path = os.path.join(DIR, person)
        labele = p.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv .imread(img_path)
            gray = cv .cvtColor(img_array, cv.COLOR_BGR2GRAY)
            face_reac = harr.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in face_reac:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                lables.append(labele)


create_train()
print("########### Training done ########### ")

#  convert to numpy
fee = np.array(features, dtype='object')
laa = np.array(lables)

# testing the cv is working
# print(f'Length of the features = {len(features)}')
# print(f'Length of the lables = {len(lables)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on labels and featuer

face_recognizer.train(fee, laa)
face_recognizer.save('face_trained.yml')
np.save('features.npy', fee)
np.save('lables.npy', laa)
