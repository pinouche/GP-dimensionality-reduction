import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from operator import itemgetter

from autoencoder import nn_autoencoder
from autoencoder import get_hidden_layers


# plot the lower-dimensional representation
def plot_low_dim(low_dim_representation, data_y, name):
    colors = np.array(["darkgray", "pink", "darkgreen", "darkblue", "darkorange", "firebrick", "black"])

    # only plot for datasets that have at most 7 classes (otherwise it's hard to visualize)
    if len(np.unique(data_y)) <= len(colors):
        plt.figure(figsize=(5, 5))
        plt.scatter(low_dim_representation[0], low_dim_representation[1], c=colors[data_y], s=10)
        plt.xlabel("First dimension", fontsize=16)
        plt.ylabel("Second dimension", fontsize=16)
        plt.grid()
        plt.savefig(name + ".pdf", dpi=600, bbox_inches='tight')
        plt.show()


def shuffle_data(x_data, y_data, seed):
    np.random.seed(seed)
    shuffle_list = np.arange(x_data.shape[0])
    np.random.shuffle(shuffle_list)
    x_data = x_data[shuffle_list]
    y_data = y_data[shuffle_list]

    return x_data, y_data


def k_fold_valifation_accuracy_rf(data_x, data_y, seed, n_splits=10):
    accuracy_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
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

        accuracy = accuracy_score(y_val, predictions)
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


def get_lower_dim(x_data, seed, low_dim=2, method="pca", x_val=None):
    scaler = StandardScaler()
    scaler = scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    if method == "nn":
        x_val = scaler.transform(x_val)

    if method == "pca":
        est = PCA(n_components=low_dim)
        est.fit(x_data)
        x_data = est.transform(x_data)

    elif method == "nn":
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        est = nn_autoencoder(seed, x_data.shape[1], low_dim)

        est.fit(x_data, x_data, batch_size=16, epochs=200, verbose=0, validation_data=(x_val, x_val), callbacks=[early_stop_callback])

        dataset = np.vstack((x_data, x_val))
        list_activations = get_hidden_layers(est, dataset)
        x_data = list_activations[3]

    else:
        raise ValueError('the dimensionality reduction method is not defined')

    return x_data, est




