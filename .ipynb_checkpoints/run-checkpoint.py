import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

df = pd.read_csv('eye_movements.csv')
y = df['label']
X = df.drop(columns=['label', 'lineNo', 'assgNo'])
from sklearn.model_selection import train_test_split
X_a, X_test, y_a, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
X_train, X_valid, y_train, y_valid = train_test_split(X_a, y_a, test_size=0.2, random_state=51)
def fit_ml_algo(algo, X_train, y_train, cv):
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    train_pred = model_selection.cross_val_predict(algo, X_valid, y_valid, cv=cv, n_jobs=-1)
    acc_cv = round(sklearn.metrics.accuracy_score(y_valid, train_pred) * 100, 2)
    return train_pred, acc, acc_cv

import tensorflow as tf
from tensorflow import keras
from keras import layers

# y_new = tf.keras.utils.to_categorical(y, num_classes=3)

model = tf.keras.Sequential([
                             tf.keras.layers.Dense(100, activation='relu'),
                             tf.keras.layers.Dense(15, activation='relu'),
                             tf.keras.layers.Dense(3, activation='relu')
])
model.compile(keras.optimizers.Adam(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=1)