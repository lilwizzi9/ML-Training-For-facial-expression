'''
https://pysource.com/2019/03/12/face-landmarks-detection-opencv-with-python/

'''



import cv2
import numpy as np
import dlib
from random import randrange


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("out.avi", fourcc, 29.6,(352,640))

color=[]
for i in range(70):
    color.append((0,30*i,0))
    #(randrange(255), randrange(255), randrange(255))


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, color[n], -1)

    out.write(frame)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
