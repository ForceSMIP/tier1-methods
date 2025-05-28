import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax, Flatten, Concatenate, Reshape, Layer
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import Model
# from tensorflow_probability import layers as tfpl
# import tensorflow_probability as tfp
# import keras_tuner as kt
K = keras.backend


### Sampling class for variational node
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    
# pattern correlation for training output
def r_squared_metric(y_true,y_pred):
    ss_res = K.sum(K.square(y_true-y_pred))
    ss_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )

def build_encoder(Xtrain, settings):
    # Xtrain has dimensions: [sample x lat x lon x variable]
    # input layer will have dimensions: [lat x lon x variable]
    input_layer = Input(shape=Xtrain.shape[1:]) 
    lays = Flatten()(input_layer)
    
    first_lay = True
    # read experiment settings to get architecture (nodes, activation)
    for hidden in settings["encoding_nodes"]:
        if first_lay:
            lays = Dense(hidden, activation=settings["activation"],
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=settings["ridge"]),
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]))(lays)
            first_lay = False
        else:
            lays = Dense(hidden, activation=settings["activation"],
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]))(lays)

    # latent space nodes
    code = Dense(settings["code_nodes"])(lays)
    # code_log_var = Dense(settings["code_nodes"])(lays)
    # code = Sampling()([code_mean, code_log_var])

    encoder = Model(inputs = [input_layer],
                outputs = [code],
                name = "encoder")
    
    return encoder, input_layer, code
    

def build_decoder(Xtrain, Ttrain, settings):
    # Xtrain has dimensions: [sample x lat x lon x variable]
    # Ttrain has dimensions: [sample x lat x lon x 1]
    # input layer will have dimensions: [code nodes], the latent space
    # output layer will have dimensions: [lat x lon x variable]
    input_layer = Input(shape=(settings['code_nodes']))
    lays = Layer()(input_layer)

    # read experiment settings to get architecture (nodes, activation)
    for hidden in settings["encoding_nodes"][::-1]:
        lays = Dense(hidden, activation=settings["activation"],
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                    bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]),
                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]))(lays)
        
    lays = Dense(np.prod(Ttrain.shape[1:]), activation='linear')(lays)
    # output is prediction
    output_layer = Reshape(Ttrain.shape[1:])(lays)

    decoder = Model(inputs = [input_layer],
                    outputs = [output_layer],
                    name = 'decoder')
    
    return decoder
    

def build_VED(Xtrain, Ttrain, settings):
    # combine encoder and decoder for full VED model and build model

    # get the encoder and decoder models
    encoder, input_layer, code = build_encoder(Xtrain, settings)
    decoder = build_decoder(Xtrain, Ttrain, settings)
    # set up the VED model: encoder -> decoder
    code = encoder(input_layer)
    reconstruction = decoder(code)
    ved = Model(inputs=[input_layer], outputs=[reconstruction], name='VED')
    # latent_loss = -0.5 * K.sum(1 + code_log_var - K.exp(code_log_var) - K.square(code_mean), axis=-1)
    # ved.add_loss(K.mean(latent_loss * settings['variational_loss']))

    return ved, encoder, decoder


def train_VED(Xtrain, Ttrain, Xval, Tval, settings,):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    # build model
    ved, encoder, decoder = build_VED(Xtrain, Ttrain, settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    # metrics to print while training
    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    # train model
    ved.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    ved.fit(Xtrain, Ttrain,
          epochs=settings["max_epochs"],
          validation_data=(Xval, Tval),
          callbacks=[early_stopping])
    
    return ved, encoder, decoder
