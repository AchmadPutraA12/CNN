from sklearn.metrics import accuracy_score

from Model import Model_CNN_IR
from Dataset import load_storage

if __name__ == '__main__':
    #objek dataset
    load = load_storage()
    print("shape x_train", load.x_train.shape)
    print("shape y_train", load.y_train.shape)

    #model ML
    ML = Model_CNN_IR(load.x_train, load.y_train, load.x_test, load.y_test)
    ML.datagenerate()
    ML.create_architecture()
    ML.train_model()
    ML.model_summary()
    ML.plot_Training()

    #analisa model
    #akurasi terhadap data training
    pred_train = ML.ModelPredict(load.x_train)
    accuracy_train = accuracy_score(load.y_train, pred_train)
    print("---------------------------------------------")
    print("akurasi terhadap data training = ", accuracy_train)

    #akurasi terhadap data test
    pred_test = ML.ModelPredict(load.x_test)
    accuracy_test = accuracy_score(load.y_test, pred_test)
    print("akurasi terhadap data test = ", accuracy_test)
    print("---------------------------------------------")