import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams["figure.figsize"] = (6, 4)
plt.style.use("ggplot")
import tensorflow as tf
from tensorflow import data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import mae
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class MLDataPrepApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Projekt')
        self.root.geometry('400x150')
        self.file_path = ''  # Zmienna na ścieżkę pliku

        self.create_widgets()

    def create_widgets(self):
        self.choose_file_button = ttk.Button(self.root, text='Wybierz plik', command=self.choose_file)
        self.choose_file_button.pack(pady=10)

        self.fill_method_label = ttk.Label(self.root, text='Wybierz metodę uzupełniania:')
        self.fill_method_label.pack()

        self.fill_method = tk.StringVar()
        self.fill_method_combobox = ttk.Combobox(self.root, textvariable=self.fill_method)
        self.fill_method_combobox['values'] = ('Średnia', 'Mediana', 'Najczęstsza wartość')
        self.fill_method_combobox.pack(pady=5)

        self.run_analysis_button = ttk.Button(self.root, text='Uruchom analizę', command=self.run_analysis)
        self.run_analysis_button.pack(pady=10)

    def choose_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path = file_path  # Przechowaj wybraną ścieżkę pliku
            messagebox.showinfo("Wybrany plik", f"Ścieżka do pliku: {file_path}")
        else:
            messagebox.showerror("Błąd", "Nie wybrano pliku.")

    def run_analysis(self):
        if not self.file_path:
            messagebox.showerror("Błąd", "Nie wybrano pliku.")
            return

        chosen_method = self.fill_method.get()
        if not chosen_method:
            messagebox.showerror("Błąd", "Nie wybrano metody uzupełniania.")
            return

        gpus = tf.config.list_physical_devices('GPU')
        gpus

        normal_df = pd.read_csv(self.file_path).iloc[:, :-1]
        anomaly_df = pd.read_csv("ptbdb_abnormal.csv").iloc[:, :-1]

        print("Shape of Normal data", normal_df.shape)
        print("Shape of Abnormal data", anomaly_df.shape)


        def plot_sample(normal, anomaly):
            index = np.random.randint(0, len(normal_df), 2)

            fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
            ax[0].plot(normal.iloc[index[0], :].values, label=f"Case {index[0]}")
            ax[0].plot(normal.iloc[index[1], :].values, label=f"Case {index[1]}")
            ax[0].legend(shadow=True, frameon=True, facecolor="inherit", loc=1, fontsize=9)
            ax[0].set_title("Normal")

            ax[1].plot(anomaly.iloc[index[0], :].values, label=f"Case {index[0]}")
            ax[1].plot(anomaly.iloc[index[1], :].values, label=f"Case {index[1]}")
            ax[1].legend(shadow=True, frameon=True, facecolor="inherit", loc=1, fontsize=9)
            ax[1].set_title("Anomaly")

            plt.tight_layout()
            plt.show()

        plot_sample(normal_df, anomaly_df)

        CLASS_NAMES = ["Normal", "Anomaly"]

        normal_df_copy = normal_df.copy()
        anomaly_df_copy = anomaly_df.copy()
        print(anomaly_df_copy.columns.equals(normal_df_copy.columns))

        normal_df_copy = normal_df_copy.set_axis(range(1, 188), axis=1)
        anomaly_df_copy = anomaly_df_copy.set_axis(range(1, 188), axis=1)
        normal_df_copy = normal_df_copy.assign(target = CLASS_NAMES[0])
        anomaly_df_copy = anomaly_df_copy.assign(target = CLASS_NAMES[1])

        df = pd.concat((normal_df_copy, anomaly_df_copy))

        normal_df.drop("target", axis=1, errors="ignore", inplace=True)
        normal = normal_df.to_numpy()
        anomaly_df.drop("target", axis=1, errors="ignore", inplace=True)
        anomaly = anomaly_df.to_numpy()

        X_train, X_test = train_test_split(normal, test_size=0.15, random_state=45, shuffle=True)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, anomaly shape: {anomaly.shape}")

        tf.keras.utils.set_random_seed(1024)

        class AutoEncoder(Model):
            def __init__(self, input_dim, latent_dim):
                super(AutoEncoder, self).__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim

                self.encoder = tf.keras.Sequential([
                    layers.Input(shape=(input_dim,)),
                    layers.Reshape((input_dim, 1)),  # Reshape to 3D for Conv1D
                    layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(2, padding="same"),
                    layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(2, padding="same"),
                    layers.Conv1D(latent_dim, 3, strides=1, activation='relu', padding="same"),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(2, padding="same"),
                ])
                # Previously, I was using UpSampling. I am trying Transposed Convolution this time around.
                self.decoder = tf.keras.Sequential([
                    layers.Conv1DTranspose(latent_dim, 3, strides=1, activation='relu', padding="same"),
        #             layers.UpSampling1D(2),
                    layers.BatchNormalization(),
                    layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
        #             layers.UpSampling1D(2),
                    layers.BatchNormalization(),
                    layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
        #             layers.UpSampling1D(2),
                    layers.BatchNormalization(),
                    layers.Flatten(),
                    layers.Dense(input_dim)
                ])

            def call(self, X):
                encoded = self.encoder(X)
                decoded = self.decoder(encoded)
                return decoded


        input_dim = X_train.shape[-1]
        latent_dim = 32

        model = AutoEncoder(input_dim, latent_dim)
        model.build((None, input_dim))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")
        model.summary()

        epochs = 100
        batch_size = 128
        early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)


        history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.1, callbacks=[early_stopping])

        plt.plot(history.history['loss'], label="Training loss")
        plt.plot(history.history['val_loss'], label="Validation loss", ls="--")
        plt.legend(shadow=True, frameon=True, facecolor="inherit", loc="best", fontsize=9)
        plt.title("Training loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

        train_mae = model.evaluate(X_train, X_train, verbose=0)
        test_mae = model.evaluate(X_test, X_test, verbose=0)
        anomaly_mae = model.evaluate(anomaly_df, anomaly_df, verbose=0)

        print("Training dataset error: ", train_mae)
        print("Testing dataset error: ", test_mae)
        print("Anormaly dataset error: ", anomaly_mae)

        def predict(model, X):
            pred = model.predict(X, verbose=False)
            loss = mae(pred, X)
            return pred, loss

        _, train_loss = predict(model, X_train)
        _, test_loss = predict(model, X_test)
        _, anomaly_loss = predict(model, anomaly)
        threshold = np.mean(train_loss) + np.std(train_loss) # Setting threshold for distinguish normal data from anomalous data

        bins = 40
        plt.figure(figsize=(9, 5), dpi=100)
        sns.histplot(np.clip(train_loss, 0, 0.5), bins=bins, kde=True, label="Train Normal")
        sns.histplot(np.clip(test_loss, 0, 0.5), bins=bins, kde=True, label="Test Normal")
        sns.histplot(np.clip(anomaly_loss, 0, 0.5), bins=bins, kde=True, label="anomaly")

        ax = plt.gca()  # Get the current Axes
        ylim = ax.get_ylim()
        plt.vlines(threshold, 0, ylim[-1], color="k", ls="--")
        plt.annotate(f"Threshold: {threshold:.3f}", xy=(threshold, ylim[-1]), xytext=(threshold+0.009, ylim[-1]),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)
        plt.legend(shadow=True, frameon=True, facecolor="inherit", loc="best", fontsize=9)
        plt.show()

        def plot_examples(model, data, ax, title):
            pred, loss = predict(model, data)
            ax.plot(data.flatten(), label="Actual")
            ax.plot(pred[0], label = "Predicted")
            ax.fill_between(range(1, 188), data.flatten(), pred[0], alpha=0.3, color="r")
            ax.legend(shadow=True, frameon=True,
                      facecolor="inherit", loc=1, fontsize=7)
        #                bbox_to_anchor = (0, 0, 0.8, 0.25))

            ax.set_title(f"{title} (loss: {loss[0]:.3f})", fontsize=9.5)

        fig, axes = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(12, 6))
        random_indexes = np.random.randint(0, len(X_train), size=5)

        for i, idx in enumerate(random_indexes):
            data = X_train[[idx]]
            plot_examples(model, data, ax=axes[0, i], title="Normal")

        for i, idx in enumerate(random_indexes):
            data = anomaly[[idx]]
            plot_examples(model, data, ax=axes[1, i], title="anomaly")

        plt.tight_layout()
        fig.suptitle("Sample plots (Actual vs Reconstructed by the CNN autoencoder)", y=1.04, weight="bold")
        fig.savefig("autoencoder.png")
        plt.show()

        def evaluate_model(model, data):
            pred, loss = predict(model, data)
            if id(data) == id(anomaly):
                accuracy = np.sum(loss > threshold)/len(data)
            else:
                accuracy = np.sum(loss <= threshold)/len(data)
            return f"Accuracy: {accuracy:.2%}"

        print("Training", evaluate_model(model, X_train))
        print("Testing", evaluate_model(model, X_test))
        print("Anomaly", evaluate_model(model, anomaly))


        def prepare_labels(model, train, test, anomaly, threshold=threshold):
            ytrue = np.concatenate((np.ones(len(X_train) + len(X_test), dtype=int), np.zeros(len(anomaly), dtype=int)))
            _, train_loss = predict(model, train)
            _, test_loss = predict(model, test)
            _, anomaly_loss = predict(model, anomaly)
            train_pred = (train_loss <= threshold).numpy().astype(int)
            test_pred = (test_loss <= threshold).numpy().astype(int)
            anomaly_pred = (anomaly_loss < threshold).numpy().astype(int)
            ypred = np.concatenate((train_pred, test_pred, anomaly_pred))

            return ytrue, ypred

        def plot_confusion_matrix(model, train, test, anomaly, threshold=threshold):
            ytrue, ypred = prepare_labels(model, train, test, anomaly, threshold=threshold)
            accuracy = accuracy_score(ytrue, ypred)
            precision = precision_score(ytrue, ypred)
            recall = recall_score(ytrue, ypred)
            f1 = f1_score(ytrue, ypred)
            print(f"""\
                Accuracy: {accuracy:.2%}
                Precision: {precision:.2%}
                Recall: {recall:.2%}
                f1: {f1:.2%}\n
                """)

            cm = confusion_matrix(ytrue, ypred)
            cm_norm = confusion_matrix(ytrue, ypred, normalize="true")
            data = np.array([f"{count}\n({pct:.2%})" for count, pct in zip(cm.ravel(), cm_norm.ravel())]).reshape(cm.shape)
            labels = ["Anomaly", "Normal"]

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=data, fmt="", xticklabels=labels, yticklabels=labels)
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.title("Confusion Matrix", weight="bold")
            plt.tight_layout()

        plot_confusion_matrix(model, X_train, X_test, anomaly, threshold=threshold)

        ytrue, ypred = prepare_labels(model, X_train, X_test, anomaly, threshold=threshold)
        print(classification_report(ytrue, ypred, target_names=CLASS_NAMES))

if __name__ == "__main__":
    root = tk.Tk()
    app = MLDataPrepApp(root)
    root.mainloop()