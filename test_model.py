#!/usr/bin/python3
#coding=utf-8

import sys

import numpy as np

from keras.utils import np_utils
from keras.models import load_model

img_rows, img_cols = 128, 128

def testModel(model, X_test, Y_test):
    classes = model.predict_classes(X_test, verbose=0)
    acc = np.mean(np.equal(Y_test, classes))
    print("accuracy: ", acc)

if __name__ == '__main__':
    if len(sys.argv) == 2 and (sys.argv[1] == 'e' or sys.argv[1] == 'f'):
        if sys.argv[1] == 'e':
            model = load_model('./model_e_0.69.h5')
            r = np.load('./ET_data_label.npz')
            X_test = r['arr_0']
            Y_test = r['arr_1']
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            testModel(model, X_test, Y_test)
        else:
            model = load_model('./model_f_1.00.h5')
            r = np.load('./IT_data_label.npz')
            X_test = r['arr_0']
            Y_test = r['arr_1']
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            testModel(model, X_test, Y_test)
    else:
        print("Use ./test_model.py f/e.")