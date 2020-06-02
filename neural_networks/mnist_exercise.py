#%% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import math
from scipy import stats
import os
from IPython.display import display

#%%
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def log_series(ini=10**(-5), end=10, n=500, base=1.5):
    log_series = np.logspace(ini, end, num=n, endpoint=True, base=base)
    return (log_series - min(log_series) + ini) / max(log_series) * end


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="tanh"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
                  optimizer=optimizer)
    return model

def describe_Xy_data(X_train,y_train,X_test,y_test,X_valid=None,y_valid=None):
    print('DATA DESCRIPTION')
    print('X_train: ' + str(X_train.shape))
    print('y_train: ' + str(y_train.shape))
    print('y range: ' + str(min(y_train)) +' - ' + str(max(y_train)))
    print('X_test : ' + str(X_test.shape))
    print('y_test : ' + str(y_test.shape))
    if X_valid.all() is not None:
        print('X_valid: ' + str(X_valid.shape))
    if y_valid.all() is not None:
        print('y_valid: ' + str(y_valid.shape))
    
K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

#%% LOAD DATA
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

#%% EXPLORE DATA
describe_Xy_data(X_train,y_train,X_test,y_test,X_valid,y_valid)

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_train[index], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

#%% CREATE MODEL
keras.backend.clear_session()
model = build_model(2,300,input_shape=[28,28])
display(keras.utils.plot_model(model, "mnist_model.png", show_shapes=True))

#%% LEARNING RATE
keras.backend.clear_session()
expon_lr = ExponentialLearningRate(factor=1.005)

history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])

#%%
plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.xlabel("Learning rate")
plt.ylabel("Loss")

#%% TRAIN with CALLBACKS
run_logdir = get_run_logdir()
checkpoint_save_cb  = keras.callbacks.ModelCheckpoint("mnist_model.h5", save_best_only=True)
early_stop_cb       = keras.callbacks.EarlyStopping(patience=10)
tensorboard_cb      = keras.callbacks.TensorBoard(run_logdir)

model = build_model(2,300,input_shape=[28,28],learning_rate=1e-1)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_save_cb, early_stop_cb, tensorboard_cb])


#%% LOAD AND TEST
model_loaded = keras.models.load_model("mnist_model.h5") # rollback to best model
model_loaded.evaluate(X_test, y_test)