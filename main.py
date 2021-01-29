# from concreteFS import ConcreteAutoencoderFeatureSelector

import numpy as np
import matplotlib.pyplot as plt
from utils import ConcreteAutoencoderFeatureSelector
from utils import interpolate_model, interpolate_predict, interpolate_train
from utils import SRCNN_train, SRCNN_predict
from utils import load_channel
from utils import unif_ind


def FS_SR() : # feature selection and super resolution 
    print("this is new")
    num_pilots = 48
    SNR = 12  # 12 or 22
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel(num_pilots, SNR)
    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    num_epochs = 1
    batch_size = 128
    learning_rate = 0.001

    ### ------ Train the Concrete Autoencoder to find the best set of pilots ----- ######

    selector = ConcreteAutoencoderFeatureSelector(K=num_pilots, output_function = interpolate_model, num_epochs=num_epochs)

    selector.fit(x_train, y_train, x_test, y_test)
    selected_indice = selector.get_support(indices=True)
    print(selected_indice)



    ### ------ Train the regressor based on designed set of pilots ----- ######
   
    train_data_1, train_label_1 = x_train[:, selected_indice], y_train
    val_data_1, val_label_1 = x_val[:, selected_indice], y_val
    test_data_1, test_label_1 = x_test[:, selected_indice], y_test
    interpolate_train(train_data_1, train_label_1, val_data_1, val_label_1, num_epochs, batch_size, learning_rate, num_pilots , SNR, '_concrete')
    predicted_concrete_train, mse_concrete_train = interpolate_predict(train_data_1, train_label_1, num_pilots, SNR, '_concrete')
    predicted_concrete_test, mse_concrete_test = interpolate_predict(test_data_1, test_label_1, num_pilots, SNR, '_concrete')
    predicted_concrete_val, mse_concrete_val = interpolate_predict(val_data_1, val_label_1, num_pilots, SNR, '_concrete')
    
    SRCNN_train(predicted_concrete_train.reshape(predicted_concrete_train.shape[0],72,14,1) ,train_label_1.reshape(train_label_1.shape[0], 72,14,1),
              predicted_concrete_val.reshape(predicted_concrete_val.shape[0],72,14,1) ,val_label_1.reshape(val_label_1.shape[0], 72,14,1) , num_epochs ,  num_pilots , SNR , '_concrete')
    SR_pred_concrete , mse_concrete_sr = SRCNN_predict(predicted_concrete_test.reshape(predicted_concrete_test.shape[0],72,14,1) , 
                                                       test_label_1.reshape(test_label_1.shape[0], 72,14,1), num_pilots, SNR, '_concrete')
    

    ### ------ Train the regressor based on uniform pilot distribution ----- ######
    train_data_2, train_label_2 = x_train[:, unif_ind(num_pilots)], y_train
    val_data_2, val_label_2 = x_val[:, unif_ind(num_pilots)], y_val
    test_data_2, test_label_2 = x_test[:, unif_ind(num_pilots)], y_test
    interpolate_train(train_data_2, train_label_2, val_data_2, val_label_2, num_epochs, batch_size, learning_rate, num_pilots, SNR, '_unif')
    predicted_unif_train, mse_unif_train = interpolate_predict(train_data_2, train_label_2, num_pilots, SNR, '_unif')
    predicted_unif_test, mse_unif_test = interpolate_predict(test_data_2, test_label_2, num_pilots, SNR, '_unif')
    predicted_unif_val, mse_unif_val = interpolate_predict(val_data_2, val_label_2, num_pilots, SNR, '_unif')
    
    SRCNN_train(predicted_unif_train.reshape(predicted_unif_train.shape[0],72,14,1) ,train_label_2.reshape(train_label_2.shape[0],72,14,1),
               predicted_unif_val.reshape(predicted_unif_val.shape[0],72,14,1) , val_label_2.reshape(val_label_2.shape[0],72,14,1) , num_epochs, num_pilots , SNR , '_unif')
    SR_pred_unif , mse_unif_sr =SRCNN_predict(predicted_unif_test.reshape(predicted_unif_test.shape[0],72,14,1) ,
                                              test_label_2.reshape(test_label_2.shape[0],72,14,1), num_pilots, SNR, '_unif')
    
    print('concrete mse: ', mse_concrete_sr)
    print('unif mse : ', mse_unif_sr)

    n = 4
    fig = plt.figure(figsize=(10, 10))
    for i in range(n) :
        ax = fig.add_subplot(n, 3, 3 * i + 1)
        #pred1 = predicted_concrete[i + 10, :].reshape([72, 14])
        pred1 = SR_pred_concrete[i+10 , :].squeeze()
        ax.imshow(pred1)

        ax = fig.add_subplot(n, 3, 3 * i + 2)
        #pred2 = predicted_unif[i + 10, :].reshape([72, 14])
        pred2 = SR_pred_unif[i+10 , :].squeeze()
        ax.imshow(pred2)

        ax = fig.add_subplot(n, 3, 3 * i + 3)
        X_label = test_label_1[i + 10, :].reshape([72, 14])
        ax.imshow(X_label)
 
  
if __name__ == '__main__' :
    FS_SR()