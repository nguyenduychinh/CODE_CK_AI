import cv2
import numpy as np
from keras.models import load_model

model = load_model('Gendent detection.h5')
path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

count = 0
classes = ['Gioi tinh: Nam','Gioi tinh:Nữ']
while True:
    Threshold= 5
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    if len(faces) == 0:
        status = 'no face'
    else:
        for x,y,w,h in faces:
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            faces = faceCascade.detectMultiScale (roi_gray)
            for (ex, ey, ew, eh) in faces:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]
                new_array = cv2.resize(face_roi, (150, 150))
                X_input = np.array(new_array).reshape(-1,150, 150, 3).astype('float64')
                Predict=np.argmax(model.predict(X_input),axis = -1)
                status = classes[int(Predict)]
            

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,status,(100, 100),font,3,(0, 255, 0),2,cv2.LINE_AA)
    cv2.imshow('Drowsiness Detection Tutorial', frame)
    if cv2.waitKey(2) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()