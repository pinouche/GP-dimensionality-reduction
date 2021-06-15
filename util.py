import numpy as np
from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from autoencoder import nn_autoencoder


def k_fold_valifation_accuracy_rf(x_train, x_test, y_train, y_test):

    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    predictions_train = classifier.predict(x_train)
    predictions_test = classifier.predict(x_test)

    accuracy_train = balanced_accuracy_score(y_train, predictions_train)
    accuracy_test = balanced_accuracy_score(y_test, predictions_test)

    return accuracy_train, accuracy_test


def compute_pareto(data):
    indice_list = np.expand_dims(np.arange(0, data.shape[0], 1), 1)
    data = np.hstack((data, indice_list))

    sorted_data = sorted(data, key=itemgetter(0, 1), reverse=True)
    pareto_idx = list()
    pareto_idx.append(0)

    cutt_off_fitness = sorted_data[0][0]
    cutt_off_length = sorted_data[0][1]

    for i in range(1, len(sorted_data)):
        if sorted_data[i][0] > cutt_off_fitness or sorted_data[i][1] < cutt_off_length:
            pareto_idx.append(i)
            if sorted_data[i][0] > cutt_off_fitness:
                cutt_off_fitness = sorted_data[i][0]
            else:
                cutt_off_length = sorted_data[i][1]

    pareto_idx = np.array(sorted_data)[:, -1][pareto_idx]

    return np.array(sorted_data)[:, :-1], pareto_idx.astype(int)


def train_base_model(train_data_x, train_data_x_pca, seed, low_dim=2):

    est = nn_autoencoder(seed, train_data_x.shape[1], train_data_x_pca.shape[1], low_dim)
    est.fit(train_data_x, train_data_x_pca, batch_size=32, epochs=200, verbose=0, validation_data=(train_data_x, train_data_x_pca))

    return est

