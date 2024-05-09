import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class convolutionalNeuralNetwork:
    def __init__(self, input_shape, num_filters, kernel_size, pool_size, dropout_rate, dense_neurons, learning_rate):
        self.model = self._build_model(input_shape, num_filters, kernel_size, pool_size, dropout_rate, dense_neurons, learning_rate)

    def _build_model(self, input_shape, num_filters, kernel_size, pool_size, dropout_rate, dense_neurons, learning_rate):
        model = Sequential()
        for i, filters in enumerate(num_filters):
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
            else:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=pool_size))
            model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(dense_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

data_path = 'ekg_dataset.csv'
df = pd.read_csv(data_path)

df = shuffle(df, random_state=42)

index_etkiet = 0
y = df.iloc[:, index_etkiet].values
X = df.drop(df.columns[index_etkiet], axis=1).values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = np.expand_dims(X, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

input_shape = (X_train.shape[1], 1)
num_filters = [64, 128]
kernel_size = 5
pool_size = 2
dropout_rate = 0.5
dense_neurons = 128
learning_rate = 0.001

detector = convolutionalNeuralNetwork(input_shape, num_filters, kernel_size, pool_size, dropout_rate, dense_neurons, learning_rate)
detector.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

# Ewaluacja modelu i przypisanie wyników do zmiennej
evaluation_results = detector.evaluate(X_test, y_test)

# Wyświetlenie wyników
# Pierwszy element to wartość straty (loss), drugi to dokładność (accuracy)
print(f"Loss: {evaluation_results[0]}")
print(f"Accuracy: {evaluation_results[1]}")
