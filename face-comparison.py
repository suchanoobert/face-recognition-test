import re
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(
    'Data/data/haarcascade_frontalface_alt2.xml')


cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=2, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        img_item = "image-1.png"
        cv2.imwrite(img_item, roi_gray)

        img1 = cv2.imread('image-1.png')
        img2 = cv2.imread('Pictures/Amogh/10.jpg')
        result = DeepFace.verify(img1, img2)
        print("Is same face: ", result["verified"])
    cv2.imshow('imshow', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# img1 = cv2.imread('Pictures/Amogh/11.jpg')
# img2 = cv2.imread('Pictures/Amogh/10.jpg')
# result = DeepFace.verify(img1, img2)
# print("Is same face: ", result["verified"])
