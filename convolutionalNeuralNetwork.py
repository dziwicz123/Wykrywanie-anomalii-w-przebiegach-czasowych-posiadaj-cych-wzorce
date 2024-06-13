import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
import json

class convolutionalNeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    @staticmethod
    def estimate_parameters(data):
        num_samples, num_features = data.shape
        base_filter_count = min(64, num_features // 8)
        num_filters = [base_filter_count * (2 ** i) for i in range(3)]
        dropout_rate = 0.05 if num_samples > 1000 else 0.1
        dense_neurons = max(256, num_features // 2)
        kernel_size = 7 if num_features > 200 else 5
        pool_size = 3
        learning_rate = 0.001

        return num_filters, kernel_size, pool_size, dropout_rate, dense_neurons, learning_rate

    def _build_model(self):
        placeholder_data = np.random.rand(1, self.input_shape[0])
        num_filters, kernel_size, pool_size, dropout_rate, dense_neurons, learning_rate = self.estimate_parameters(placeholder_data)

        self.model = Sequential()
        for i, filters in enumerate(num_filters):
            if i == 0:
                self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=self.input_shape))
            else:
                self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            self.model.add(MaxPooling1D(pool_size=pool_size))
            self.model.add(Dropout(dropout_rate))

        self.model.add(Flatten())
        self.model.add(Dense(dense_neurons, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.model = load_model(file_path)
        print("Model loaded successfully")

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32):
        early_stopping = EarlyStopping(monitor='accuracy', min_delta=0.005, patience=3, restore_best_weights=True)
        return self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[early_stopping])

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

def save_label_mapping(label_encoder, output_path):
    labels = label_encoder.classes_
    num_classes = len(labels)
    label_mapping = {}
    for i, label in enumerate(labels):
        binary_vector = [0] * num_classes
        binary_vector[i] = 1
        label_mapping[label] = binary_vector

    with open(output_path, 'w') as json_file:
        json.dump(label_mapping, json_file, indent=4)

if __name__ == "__main__":
    data_path = 'ekg_dataset.csv'
    df = pd.read_csv(data_path)
    df = shuffle(df, random_state=42)

    label_index = 0
    y = df.iloc[:, label_index].values
    X = df.drop(df.columns[label_index], axis=1).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y))

    y = to_categorical(y, num_classes)  # Convert labels to one-hot encoding
    save_label_mapping(label_encoder, 'label_mapping.json')  # Save label mapping to JSON

    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    input_shape = (X_train.shape[1], 1)

    detector = convolutionalNeuralNetwork(input_shape, num_classes)
    detector._build_model()

    start_time = time.time()
    training_time = time.time() - start_time
    print(f"Time to build and compile the model: {training_time:.2f} seconds")

    start_time = time.time()
    detector.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    start_time = time.time()
    evaluation_results = detector.evaluate(X_test, y_test)
    eval_time = time.time() - start_time
    print(f"Evaluation time: {eval_time:.2f} seconds")

    print(f"Loss: {evaluation_results[0]}")
    print(f"Accuracy: {evaluation_results[1]}")
