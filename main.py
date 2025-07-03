import cv2
import numpy as np
import face_recognition

# get image as bgr so we need to convert to rgb first for our library

# orig image
imgCurry = face_recognition.load_image_file("images/bron.png")
imgCurry = cv2.cvtColor(imgCurry, cv2.COLOR_BGR2RGB)

# test image
imgCurryTest = face_recognition.load_image_file("images/currytest.png")
imgCurryTest = cv2.cvtColor(imgCurryTest, cv2.COLOR_BGR2RGB)

# detect face
faceLocation = face_recognition.face_locations(imgCurry)[0]
testLocation = face_recognition.face_locations(imgCurryTest)[0]
encodeCurry = face_recognition.face_encodings(imgCurry)[0]
encodeCuryTest = face_recognition.face_encodings(imgCurryTest)[0]
# print(encodeCurry)


# draw face rectangles
cv2.rectangle(imgCurry, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2) #imgCurry
cv2.rectangle(imgCurryTest, (testLocation[3], testLocation[0]), (testLocation[1], testLocation[2]), (255, 0, 255), 2) #imgCurry

# results + face distance (lower num is a match)

results = face_recognition.compare_faces([encodeCurry], encodeCuryTest)
faceDistance = face_recognition.face_distance([encodeCurry], encodeCuryTest)

print(results, faceDistance)
cv2.putText(imgCurryTest, f'{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
# show image
cv2.imshow("Curry", imgCurry)
cv2.imshow("Curry Test", imgCurryTest)
cv2.waitKey(0)

