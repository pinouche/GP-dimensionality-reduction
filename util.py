import numpy as np
from operator import itemgetter

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from autoencoder import nn_autoencoder


def k_fold_valifation_accuracy_rf(data_x, data_y, n_splits=5):
    accuracy_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_indices, val_indices in kf.split(data_x):
        x_train, y_train = data_x[train_indices], data_y[train_indices]
        x_val, y_val = data_x[val_indices], data_y[val_indices]

        scaler = StandardScaler()
        scaler = scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)

        classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_val)

        accuracy = balanced_accuracy_score(y_val, predictions)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list), np.std(accuracy_list)


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


def train_base_model(x_data, seed, low_dim=2):

    est = nn_autoencoder(seed, x_data.shape[1], low_dim)
    est.fit(x_data, x_data, batch_size=32, epochs=200, verbose=0)

    return est




