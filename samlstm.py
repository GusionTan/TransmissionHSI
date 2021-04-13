from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import optimizers

import tensorflow as tf
import src.utils as ut
import numpy as np
import matplotlib as mpl


# lstm
def lstm(X_train, y_train, X_test, y_test, xp, yp):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, input_shape=(10, 20)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    adam = optimizers.Adam(learning_rate=0.001, decay=1e-6)

    filepath = "./model/saelstm-model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, callbacks=callbacks_list)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('\nValidation loss: %.6f' % loss)
    print('Validation accuracy: %.4f' % accuracy)

    after_loss, after_acc = model.evaluate(xp, yp)
    print("test acc is %.4f:" % after_acc)

    return np.argmax(model.predict(xp), axis=-1)


# cat sam
def sam(l1, l2):
    l1 = l1.reshape(-1, 1)
    l2 = l2.reshape(-1, 1)
    return np.arccos(np.dot(l2.T, l1) / (np.linalg.norm(l2) * np.linalg.norm(l1)))


# find the first n most similar vector subscripts
def nMinNumber(m, l, n):
    temp = []
    # add self
    temp.append(m)
    l[m] = float("inf")
    for i in range(n-1):
        minindex = l.index(min(l))
        temp.append(minindex)
        l[minindex] = float("inf")

    return temp


# Generate similarity sequences
def similaritySequence(x, n):
    k1, k2 = x.shape[0], x.shape[1]
    sequence = []
    for i in range(k1):
        xlist, xmatrix = [], []

        # Find the Sam of each vector relative to the other vector
        for j in range(k1):
            xlist.append(sam(x[i], x[j]))

        # Output the first n most similar vector subscripts
        xnlist = np.array(nMinNumber(i, xlist, n))

        for z in xnlist:
            xmatrix.append(x[z]) # the shape of xmatrix is (n, pca_size)

        sequence.append(xmatrix)

    return np.array(sequence)


if __name__ == '__main__':
    x1, y1 = ut.loadData('l')  # (720, 120)
    x2, y2 = ut.loadData('h')  # (1490, 680)

    # Drawing parameters
    k1, k2 = x2.shape[0], x2.shape[1]
    draw = np.zeros((k1, k2))

    x1 = x1.reshape(x1.shape[0] * x1.shape[1], -1)
    x2 = x2.reshape(k1 * k2, -1)
    y1 = y1.reshape(-1)
    y3 = y2.reshape(-1)  # keep y2

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
    X = ut.pca(X, 20)
    x1_pca, x2_pca = X[:s1, :], X[s1:, :]
    print(x1_pca.shape, x2_pca.shape)

    # Generate input features
    mx1 = similaritySequence(x1_pca, 10) # (numbers, 10, pca_size)
    mx2 = similaritySequence(x2_pca, 10)

    # Ensuring Input Format
    mx1 = mx1.reshape(-1, 10, 20)
    mx2 = mx2.reshape(-1, 10, 20)

    Xtrain, Xtest, ytrain, ytest = train_test_split(mx1, y1-1, test_size=0.8, random_state=340, stratify=y1)
    y_pre = lstm(Xtrain, ytrain, Xtest, ytest, x2_pca, y3-1) +1

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