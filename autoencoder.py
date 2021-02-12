import keras
from sklearn.preprocessing import StandardScaler


def get_hidden_layers(model, data_x):

    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)

    def keras_function_layer(model_layer, data):
        hidden_func = keras.backend.function(model.layers[0].input, model_layer.output)
        result = hidden_func([data.astype('float32')])

        return result

    hidden_layers_list = []
    for index in range(len(model.layers)):
        if isinstance(model.layers[index], keras.layers.convolutional.Conv2D) or isinstance(model.layers[index],
                                                                                            keras.layers.Dense):
            hidden_layer = keras_function_layer(model.layers[index], data_x)
            hidden_layers_list.append(hidden_layer)

    return hidden_layers_list


def nn_autoencoder(seed, input_size, latent_size):
    initializer = keras.initializers.glorot_normal(seed=seed)

    model = keras.models.Sequential([

        keras.layers.Dense(input_size, activation="elu", use_bias=True,
                           trainable=True, kernel_initializer=initializer, input_shape=(input_size,)),

        keras.layers.Dense(int((input_size + latent_size) / 2), activation="elu", use_bias=True,
                           trainable=True, kernel_initializer=initializer),

        keras.layers.Dense(int((input_size + latent_size) / 4), activation="elu", use_bias=True,
                           trainable=True, kernel_initializer=initializer),

        # latent_layer
        keras.layers.Dense(latent_size, activation=keras.activations.linear, use_bias=False,
                           trainable=True, kernel_initializer=initializer),

        keras.layers.Dense(int((input_size + latent_size) / 4), activation="elu", use_bias=True,
                           trainable=True, kernel_initializer=initializer),

        keras.layers.Dense(int((input_size + latent_size) / 2), activation="elu", use_bias=True,
                           trainable=True, kernel_initializer=initializer),

        keras.layers.Dense(input_size, activation=keras.activations.linear, use_bias=False,
                           trainable=True, kernel_initializer=initializer)
    ])

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='mse', metrics=['mse'])

    return model
