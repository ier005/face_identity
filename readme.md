# face_identity

Identity of emotion (angry,  smile, surprise and neutral) and face, with cnn.

Identity of face got an accuracy of 100%, while identity of emotion needs to be optimized.

## Instructions

train_face.py: for training the net and record the model. (which will stop when the val_loss do not get lower in ten epochs)

test_face.py: for testing the performance of the net

loadface.py: for verifying face detection and generating npz from graphs

---

*The following is numpy matrix generated from graphs (not uploaded):*

data_label.npz: the data for training, containing the matrix of graph and the labels of emotion and identity

ET_data_label.npz: the test data for emotion identity (from the people out of the training data)

IT_data_label.npz: the test data for face identity
