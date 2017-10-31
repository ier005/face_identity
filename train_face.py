#!/usr/bin/python3
#coding=utf-8

import os
import sys

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

# input image dimensions
img_rows, img_cols = 128, 128





# for emotion detection
nb_classes = 4
epochs = 120
batch_size = 5

# number of convolutional filters to use
nb_filters1, nb_filters2 = 10, 20 #5, 10
# size of pooling area for max pooling
nb_pool = 4
# convolution kernel size
nb_conv = 3

#------------------------------------

# for face identity
inb_classes = 5
iepochs = 80
ibatch_size = 5

# number of convolutional filters to use
inb_filters1, inb_filters2 = 5, 10
# size of pooling area for max pooling
inb_pool = 4
# convolution kernel size
inb_conv = 3

#------------------------------------





def loadEmotionData():
	if os.path.exists('./data_label.npz'):
		r = np.load('./data_label.npz')
		return (r['arr_0'], r['arr_1'], r['arr_2'])
	face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
	dir = ['angry', 'neutral', 'smile', 'surprise']
	e_label = []
	i_label = []
	e_face = []
	iden = 0
	for psn in os.listdir('./pics'):
		for i in range(4):
			path = './pics/' + psn + '/' + dir[i] + '/'
			for file in os.listdir(path):
				print(path+file)
				image = cv2.imread(path+file, 0)
				faces = face_cascade.detectMultiScale(image, 1.3, 5)
				print(len(faces))
				for (x, y, w, h) in faces:
					e_face.append(cv2.resize(image[y:y+h, x:x+w], (img_rows, img_cols)))
					e_label.append(i)
					i_label.append(iden)
		iden += 1

	e_face = np.asarray(e_face)
	e_label = np.asarray(e_label)
	i_label = np.asarray(i_label)
	np.savez('./data_label.npz', e_face, e_label, i_label)
	return (e_face, e_label, i_label)



def netModel(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()
    model.add(Convolution2D(nb_filters1, (nb_conv, nb_conv),  
                            padding='valid',  
                            input_shape=(img_rows, img_cols, 1)))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))  
  
    model.add(Convolution2D(nb_filters2, (nb_conv, nb_conv)))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))  
    #model.add(Dropout(0.25))  
  
    model.add(Flatten())  
    model.add(Dense(1000)) #Full connection  
    model.add(Activation('tanh'))  
    #model.add(Dropout(0.5))  
    model.add(Dense(nb_classes))  
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def netModel_iden(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()
    model.add(Convolution2D(inb_filters1, (inb_conv, inb_conv),  
                            padding='valid',  
                            input_shape=(img_rows, img_cols, 1)))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(inb_pool, inb_pool)))  
  
    model.add(Convolution2D(inb_filters2, (inb_conv, inb_conv)))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(inb_pool, inb_pool)))  
    #model.add(Dropout(0.25))  
  
    model.add(Flatten())  
    model.add(Dense(1000)) #Full connection  
    model.add(Activation('tanh'))  
    #model.add(Dropout(0.5))  
    model.add(Dense(inb_classes))  
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def trainModel(model, X_train, Y_train):
    save_best = ModelCheckpoint('model_e_{val_acc:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    index = list(range(len(X_train)))
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[save_best, early_stop])
    return model

def trainModel_iden(model, X_train, Y_train):
    save_best = ModelCheckpoint('model_f_{val_acc:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    index = list(range(len(X_train)))
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    model.fit(X_train, Y_train, batch_size=ibatch_size, epochs=iepochs, verbose=1, validation_split=0.2, callbacks=[save_best, early_stop])
    return model




if __name__ == '__main__':
	if len(sys.argv) == 2 and (sys.argv[1] != 'e' or sys.argv[1] != 'f'):
	
		(X_train, Y_train, Y_train1) = loadEmotionData()
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		
		if sys.argv[1] == 'e':
			Y_train = np_utils.to_categorical(Y_train, nb_classes)
			model = netModel(0.001)
			trainModel(model, X_train, Y_train)
		else:	
			Y_train1 = np_utils.to_categorical(Y_train1, inb_classes)
			model = netModel_iden(0.001)
			trainModel_iden(model, X_train, Y_train1)

	else:
		print("Use ./faec.py f/e (f for face identity, e for emotion identity)")
