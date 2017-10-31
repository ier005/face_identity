#!/usr/bin/python3
#coding=utf-8

import os

import cv2
import numpy as np


# input image dimensions
img_rows, img_cols = 128, 128

def loadImage():
	face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
	dir = ['angry', 'neutral', 'smile', 'surprise']
	for psn in os.listdir('./pics_test/iden'):
		path = './pics_test/iden/' + psn + '/' 
		for file in os.listdir(path):
			print(path+file)
			image = cv2.imread(path+file, 0)
			faces = face_cascade.detectMultiScale(image, 1.3, 5)
			print(len(faces))
			if len(faces) != 1:
				os.remove(path+file)
				continue
			for (x, y, w, h) in faces:
				cv2.imshow('a', cv2.resize(image[y:y+h, x:x+w], (512, 512)))
				cv2.waitKey()

def saveImage():
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    dir = ['angry', 'neutral', 'smile', 'surprise']
    e_label = []
    i_label = []
    e_face = []
    iden = 0
    for psn in os.listdir('./pics_test/iden'):
        path = './pics_test/iden/' + psn + '/'
        for file in os.listdir(path):
            print(path+file)
            image = cv2.imread(path+file, 0)
            faces = face_cascade.detectMultiScale(image, 1.3, 5)
            print(len(faces))
            if len(faces) != 1:
                os.remove(path+file)
                continue
            for (x, y, w, h) in faces:
                e_face.append(cv2.resize(image[y:y+h, x:x+w], (img_rows, img_cols)))
                e_label.append(iden)
        iden += 1
        print(id)
    e_face = np.asarray(e_face)
    e_label = np.asarray(e_label)
    np.savez('./IT_data_label.npz', e_face, e_label)

if __name__ == '__main__':
    saveImage()