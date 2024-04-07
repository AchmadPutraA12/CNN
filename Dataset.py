import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class load_storage:
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_dir = 'newDataset/data_train'
    test_dir = 'newDataset/data_test'

    def __init__(self):
        self.loaddataTrainWithTensorFlow()
        self.loaddataTestWithTensorFlow()

    def loaddataTrainWithTensorFlow(self):
        # Load data with TensorFlow
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            image_size=(400, 600),
            color_mode='grayscale')
        for images, labels in train_data:
            self.x_train.extend(images.numpy())
            self.y_train.extend(labels.numpy())

        # Convert lists to TensorFlow variables
        self.x_train = tf.convert_to_tensor(self.x_train)
        self.y_train = tf.convert_to_tensor(self.y_train)

    def loaddataTestWithTensorFlow(self):
        # Load data with TensorFlow
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_dir,
            image_size=(400, 600),
            color_mode='grayscale')
        for images, labels in test_data:
            self.x_test.extend(images.numpy())
            self.y_test.extend(labels.numpy())

        # Convert lists to TensorFlow variables
        self.x_test = tf.convert_to_tensor(self.x_test)
        self.y_test = tf.convert_to_tensor(self.y_test)