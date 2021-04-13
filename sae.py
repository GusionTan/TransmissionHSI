from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import optimizers

import tensorflow as tf
import src.utils as ut
import numpy as np
import matplotlib as mpl
import os


# sae
def sae(X_train, y_train, X_test, y_test, xp, yp):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    adam = optimizers.Adam(learning_rate=0.001, decay=1e-6)

    filepath = "./model/sae-model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, callbacks=callbacks_list)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('\nValidation loss: %.6f' % loss)
    print('Validation accuracy: %.4f' % accuracy)

    after_loss, after_acc = model.evaluate(xp, yp)
    print("test acc is %.4f:" % after_acc)

    return np.argmax(model.predict(xp), axis=-1)


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
                patch_list.append(whole_m[i - k: i + k + 1, j - k: j + k + 1].reshape(-1))

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
    # generate m1
    m1_1, m1_2 = X[:s1, :], X[s1:, :]

    # generate m2
    X = ut.pca(X, 10)
    x1_pca, x2_pca = X[:s1, :], X[s1:, :]
    m2_1 = refactoring(p1, p2, x1_pca, y1_, 9)
    m2_2 = refactoring(k1, k2, x2_pca, y3_, 9)
    # print(m1_1.shape, m2_1.shape, m1_2.shape, m2_2.shape)

    # combined m1 and m2
    mx1 = np.concatenate((m1_1, m2_1), axis=1)
    mx2 = np.concatenate((m1_2, m2_2), axis=1)
    print("mx1.shape: ", mx1.shape, "mx2.shape: ", mx2.shape)

    Xtrain, Xtest, ytrain, ytest = train_test_split(mx1, y1 - 1, test_size=0.8, random_state=340, stratify=y1)
    print("dataset size: ", Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

    y_pre = sae(Xtrain, ytrain, Xtest, ytest, mx2, y3 -1) + 1

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
