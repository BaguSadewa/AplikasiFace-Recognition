import cv2, time
import os
from PIL import Image
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Recognizer = cv2.face.LBPHFaceRecognizer_create()
Recognizer.read('MyData/training.xml')
a = 0
while True :
    a = a+1
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in wajah:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        id, conf = Recognizer.predict(gray[y:y+h,x:x+w])
        if (id==1):
            id = 'Bagoes'
        elif (id==2):
            id = 'fulan'
        cv2.putText(frame,str(id),(x+40,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))
    cv2.imshow ("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break
video.release()
cv2.destroyAllWindows()