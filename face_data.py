import cv2
import numpy as np

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip=0
face_data=[]
dataset_path = 'data/'
file_name = input('Enter name of the person')

while True:
    ret,cframe = cap.read()
    frame=cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue

    faces=face_cascade.detectMultiScale(frame, 1.2,5)
    faces=sorted(faces, key=lambda f: f[2]*f[3])
    skip=skip+1

    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)

        offset=10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section=cv2.resize(face_section, (100,100))

    
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        
        cv2.imshow("Video",cframe)
        cv2.imshow('Face section', face_section)

    if skip==250:
        break
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data=np.asarray(face_data)
print(face_data.shape)
face_data=face_data.reshape((face_data.shape[0],-1))

np.save(dataset_path+file_name+'.npy',face_data)

cap.release()
cv2.destroyAllWindows()
