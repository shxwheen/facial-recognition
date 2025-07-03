import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'images'
images = []
names = []
# put images in list and get names (for loop)
imgList = os.listdir(path)

for img in imgList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    names.append(os.path.splitext(img)[0])
print(names)

def findEncodings(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodings.append(encoding)
    return encodings

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}\n')





encodeListKnownFaces = findEncodings(images)
# print(len(encodeListKnownFaces))
cap = cv2.VideoCapture(0)
# while loop to get each frame one by one
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodingsOfCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodingsOfCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnownFaces, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnownFaces, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = 4*y1, 4*x2, 4*y2, 4*x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Video', img)
    cv2.waitKey(1)

