from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from matplotlib import pyplot as plt

import src.utils as ut
import numpy as np
import matplotlib as mpl
import os


def svc(X_train, y_train, X_test, y_test, xp, yp):
    """
    Use dataset 1 to train and validate the model, and use dataset 2 to test
    X_train, y_train, X_test, y_test from dataset1, xp, yp from dataset2
    """
    model = SVC(C=1e3, kernel='rbf', gamma=10, probability=True, class_weight='balanced')

    """
    # adjustable parameters
    param_grid = {'C': [1, 10, 100, 1e3, 1e4],
                'gamma': [0.01, 0.1, 1, 10, 50], } # 1e4, 0.1

    param_grid = {'C': [10, 50, 100, 200],
                 'gamma': [0.7, 1, 2, 2.5, 3], }    # 100, 5

    clf = GridSearchCV(SVC(kernel='rbf', probability=True, class_weight='balanced'), param_grid)
    
    clf.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.2f"
          % (clf.best_params_, clf.best_score_))
    """

    model.fit(X_train, y_train)
    # validate
    y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict) * 100
    print("(svm)before acc: %.4f" % accuracy)

    # test
    y_all = model.predict(xp)
    accuracy = metrics.accuracy_score(yp, y_all) * 100
    print("(svm)after acc: %.4f" % accuracy)

    return y_all


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
    X = ut.pca(X, 30)
    x1_pca, x2_pca = X[:s1, :], X[s1:, :]
    Xtrain, Xtest, ytrain, ytest = train_test_split(x1_pca, y1, test_size=0.8, random_state=340, stratify=y1)
    y_pre = svc(Xtrain, ytrain, Xtest, ytest, x2_pca, y3)

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