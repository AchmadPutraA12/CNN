import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from Model import Model_CNN_IR
from Dataset import load_storage

# Fungsi untuk menyerialisasi contoh ke format TFRecord
def serialize_example(feature, label):
    # Membuat dictionary untuk feature dan label
    feature_dict = {
        'feature': tf.train.Feature(float_list=tf.train.FloatList(value=feature.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    # Membuat example
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

# Main script
if __name__ == '__main__':
    # Objek dataset
    load = load_storage()
    print("shape x_train", load.x_train.shape)
    print("shape y_train", load.y_train.shape)

    # Model ML
    ML = Model_CNN_IR(load.x_train, load.y_train, load.x_test, load.y_test)
    ML.datagenerate()
    ML.create_architecture()
    ML.train_model()
    ML.model_summary()
    ML.plot_Training()

    # Analisa model
    # Akurasi terhadap data training
    pred_train = ML.ModelPredict(load.x_train)
    accuracy_train = accuracy_score(load.y_train, pred_train)
    print("---------------------------------------------")
    print("akurasi terhadap data training = ", accuracy_train)

    # Akurasi terhadap data uji
    pred_test = ML.ModelPredict(load.x_test)
    accuracy_test = accuracy_score(load.y_test, pred_test)
    print("akurasi terhadap data test = ", accuracy_test)
    print("---------------------------------------------")

    with tf.io.TFRecordWriter('train_data.tfrecord') as writer:
        for feature, label in zip(load.x_train, load.y_train):
            example = serialize_example(feature.numpy(), label.numpy())
            writer.write(example)
    print("Data latih disimpan dalam file 'train_data.tfrecord'")

    with tf.io.TFRecordWriter('test_data.tfrecord') as writer:
        for feature, label in zip(load.x_test, load.y_test):
            example = serialize_example(feature.numpy(), label.numpy())
            writer.write(example)
    print("Data uji disimpan dalam file 'test_data.tfrecord'")
