import numpy as np
import os
import cv2

def dist(p1, p2):
    s=0
    n=p1.shape[0]
    for i in range(0,n):
        s=s+((p1[i]-p2[i])**2)
    return s**0.5

def KNN(X,Y,im):
    k=11
    n=X.shape[0]
    arr=[]
    for i in range(0,n):
        arr.append((dist(im,X[i]),Y[i]))
    arr.sort()
    arr=arr[:k]
    arr=np.array(arr,dtype='uint32')
    arr=arr[:,1]
    arr=np.unique(arr, return_counts=True)
    index=arr[1].argmax()
    return arr[0][index]

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip=0
face_data=[]
labels=[]
dataset_path = 'data/'

class_id=0
names ={}


for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        names[class_id]=fx[:-4]
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)

        target = class_id*np.ones((data_item.shape[0]))
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels, axis=0).reshape((-1,1))

skip=0
name=" "
while True:
    skip+=1
    ret,cframe=cap.read()
    frame=cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)

    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for face in faces:
        x,y,w,h=face

        offset=10
        face_section=frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        
        if skip%20==0:
            pred=KNN(face_dataset, face_labels,face_section.flatten())
            name=names[int(pred)]

        cv2.putText(cframe, name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
        cv2.rectangle(cframe,(x,y),(x+w, y+h), (255,25,255),3)

    cv2.imshow("Faces",cframe)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
