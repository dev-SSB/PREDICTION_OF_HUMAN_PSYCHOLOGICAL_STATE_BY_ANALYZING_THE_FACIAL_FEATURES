import tkinter as tk
#from tkinter import *
from tkinter import messagebox as ms
import sqlite3
from keras.models import load_model
from PIL import Image, ImageTk
import re
import random
import os, os.path
import cv2
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

count_happy = 0
count_sad = 0
count_disgusted = 0
count_angry = 0
count_fearful = 0
count_suprised = 0
count_neutral = 0

def upload():

    global count_happy
    global count_sad
    global count_disgusted
    global count_angry
    global count_fearful
    global count_suprised
    global count_neutral

    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
        
    model.load_weights('25FebModelPlot.h5')
     # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
        # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    sampleNum = 0
        # start the webcam feed
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    while True:
            # Find haar cascade to draw bounding box around face
        ret, img = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        
    
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            print(maxindex)
            if maxindex == 2 :
                    count_fearful+=1
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/Fearful/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'Fearful', (x + w, y + h), 1, 1, (255, 0,0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame',img)
        ###############################################################################################################            #cv2.waitKey(1);
            elif maxindex == 1:
                    count_disgusted += 1
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/Disgusted/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'happy', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
                 ###############################################################################################################            #cv2.waitKey(1);
            elif maxindex == 0:
                    count_angry += 1 
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/angry/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'happy', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
                
         ###############################################################################################################            #cv2.waitKey(1);
            elif maxindex == 3:
                    count_happy += 1
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/Happy/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'happy', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
     
            elif maxindex == 4:
                    count_neutral += 1
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/Neutral/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'neutral', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame', img)
        ##########################################################################################################        #cv2.waitKey(1);
            elif maxindex == 5:
                    count_sad += 1
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/Sad/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'sad', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame', img)
                    # cv2.waitKey(1);
        ###########################################################################################################
     ###########################################################################################################        #cv2.waitKey(1);
            elif maxindex == 6:
                    count_suprised += 1
                    sampleNum = sampleNum + 1
                    cv2.imwrite("personal_dataset/Surprised/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # cv2.putText(img, 'sad', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    # cv2.waitKey(100)
                    # cv2.imshow('frame', img)
                    # cv2.waitKey(1);
                    
            cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)
        cv2.imshow('Video', cv2.resize(img,(600,600),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()    

def files_count():
    global count_happy
    global count_sad
    global count_disgusted
    global count_angry
    global count_fearful
    global count_suprised
    global count_neutral

    total_count = count_suprised + count_angry + count_disgusted + count_fearful + count_neutral + count_happy + count_sad
    
    if ((count_neutral/2)+count_happy+count_suprised) >= 0.75*total_count:
        str_label = "You were almost always confident during the interview"
    elif ((count_neutral/2)+count_happy+count_suprised) >= 0.45*total_count:
        str_label = "You were mildly confident during the interview, some more confidence would significantly imrprove your chances"
    elif ((count_neutral/2)+count_happy+count_suprised) >= 0.20*total_count:
          str_label = "Your confidence during interview felt low. You need to show more confidence during the interview"
    return "Your confidence during interview felt low. \nYou need to show more confidence during the interview"

