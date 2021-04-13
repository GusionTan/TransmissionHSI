from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mpl
import src.utils as ut
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# hybirdsn
def hybirdsn(Xtrain, ytrain, Xtest, ytest, pca_size, windowSize, xp, yp):
    output_units = 5  # output nodes
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, pca_size, 1)
    ytrain = utils.to_categorical(ytrain)

    S = windowSize

    # input layer
    input_layer = Input((S, S, pca_size, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)

    # conv3d_shape = conv_layer3._keras_shape
    # conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

    conv_layer3 = Reshape((19, 19, 18*32))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    # load weights
    model.load_weights("./model/hybirdsn-model.hdf5")

    # # compiling the model
    adam = Adam(lr=0.001, decay=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #
    # # checkpoint
    # filepath = "./model/hybirdsn-model.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    #
    # history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=10, callbacks=callbacks_list)
    #
    #
    # # print loss_image
    # plt.figure(figsize=(7,7))
    # plt.grid()
    # plt.plot(history.history['loss'])
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.legend(['Training','Validation'], loc='upper right')
    # # plt.savefig("loss_curve.pdf")
    # plt.show()

    # Validation
    Xtest = Xtest.reshape(-1, windowSize, windowSize, pca_size, 1)
    ytest = utils.to_categorical(ytest)
    xp = xp.reshape(-1, windowSize, windowSize, pca_size, 1)
    yp = utils.to_categorical(yp)

    xp_pred= model.predict(xp)
    xp_pred = np.argmax(xp_pred, axis=1)

    before_loss, before_acc = model.evaluate(Xtest, ytest)
    print("before acc: %.4f:" % before_acc)

    after_loss, after_acc = model.evaluate(xp, yp)
    print("after acc: %.4f:" % after_acc)

    return xp_pred

# generate the input features
def refactoring(n, m, x, y, window_size):
    k = int(window_size / 2)
    new_m = np.zeros((n * m, x.shape[1]))
    j = 0
    for i in range(y.shape[0]):
        if y[i] != 0:
            new_m[i] = x[j]
            j += 1

    # The PCA matrix corresponding to the original image is generated with a background of 0
    new_m = new_m.reshape(n, m, x.shape[1])
    # Peripheral fill 0
    whole_m = np.zeros((n + 2 * k, m + 2 * k, x.shape[1]))
    whole_m[k: n + k, k:m + k] = new_m
    # use sum to judge object
    sum_m = np.sum(whole_m, axis=2)

    # generate patch
    patch_list = []
    for i in range(sum_m.shape[0]):
        for j in range(sum_m.shape[1]):
            if sum_m[i, j] != 0:
                patch_list.append(whole_m[i - k: i + k + 1, j - k: j + k + 1].reshape(window_size, window_size, x.shape[1], 1))

    return np.array(patch_list)


if __name__ == '__main__':
    x1, y1 = ut.loadData('l')  # (720, 120)
    x2, y2 = ut.loadData('h')  # (1490, 680)

    # Drawing parameters and recovering m
    p1, p2, k1, k2 = x1.shape[0], x1.shape[1], x2.shape[0], x2.shape[1]
    draw = np.zeros((k1, k2))

    x1 = x1.reshape(x1.shape[0] * x1.shape[1], -1)
    x2 = x2.reshape(k1 * k2, -1)
    y1 = y1.reshape(-1)
    y3 = y2.reshape(-1)  # keep y2
    y1_ = y1  # use y1_ and y3_ create m2
    y3_ = y3

    # find a global l0
    # l0 = x1[y1 == 0, :]
    # l0 = l0[200:300, :]
    # l0 = np.mean(l0, axis=0) +1

    # identify only white areas
    x1 = x1[y1 > 0, :]
    y1 = y1[y1 > 0]
    x2 = x2[y3 > 0, :]
    y3 = y3[y3 > 0]
    # concatenate x1 and x2 do the same preprocessing
    s1 = x1.shape[0]
    s2 = x2.shape[0]
    x_total = np.concatenate((x1, x2), axis=0) + 1

    # new features
    # x_total = ut.absorptivity(l0, x_total)

    X = ut.normalization(x_total)

    # generate windows input (numbers, size, size, pca_n, 1)
    X = ut.pca(X, 30)
    x1_pca, x2_pca = X[:s1, :], X[s1:, :]
    mx1 = refactoring(p1, p2, x1_pca, y1_, 25)
    mx2 = refactoring(k1, k2, x2_pca, y3_, 25)
    print("mx1.shape, mx2.shape: ", mx1.shape, mx2.shape)


    Xtrain, Xtest, ytrain, ytest = train_test_split(mx1, y1 - 1, test_size=0.8, random_state=340, stratify=y1)
    print("dataset size: ", Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

    y_pre = hybirdsn(Xtrain, ytrain, Xtest, ytest, 30, 25, mx2, y3-1) + 1

    # visualization
    index = 0
    for i in range(k1):
        for j in range(k2):
            if y2[i, j] != 0:
                draw[i, j] = y_pre[index]
                index += 1

    colors = ['mintcream', 'tomato', 'skyblue', 'gold', 'peachpuff', 'aquamarine']
    cmap = mpl.colors.ListedColormap(colors)
    plt.subplot(1, 2, 1)
    plt.imshow(y2, cmap=cmap)
    plt.subplot(1, 2, 2)
    plt.imshow(draw, cmap=cmap)
    plt.show()