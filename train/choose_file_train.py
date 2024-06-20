from tkinter import Frame, Label, filedialog, IntVar, StringVar
from creator import create_button, create_my_text
from train.save_model import create_save_model_canvas

import numpy as np
import pandas as pd
import time

from convolutionalNeuralNetwork import convolutionalNeuralNetwork

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

variable = 0

def browse_file(canvas, text_id, filename_var):
    filename = filedialog.askopenfilename(
        title="Wybierz plik",
        filetypes=[("Pliki csv", "*.csv"), ("Wszystkie pliki", "*.*")],
    )
    if filename:
        print(f"Wybrano plik: {filename}")
        canvas.itemconfig(text_id, text=f"{filename}", font=("Arial", 14))
        filename_var.set(filename)


def change_state(canvas, text_id):
    global variable

    if variable == 0:
        canvas.itemconfig(text_id, text="X")
        variable = 1
    else:
        canvas.itemconfig(text_id, text="")
        variable = 0


def switch_to_save_model_canvas(canvas, filename_var, variable):
    filename = filename_var.get()
    print(f"Przekazywany plik: {filename}")

    df = pd.read_csv(filename)

    print(variable)
    if variable == 1:
        y = df.iloc[:, 0].values
        X = df.drop(df.columns[0], axis=1).values
    else:
        y = df.iloc[:, -1].values
        X = df.drop(df.columns[-1], axis=1).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y))

    y = to_categorical(y, num_classes)

    X = np.expand_dims(X, axis=-1)

    input_shape = (X.shape[1], 1)

    detector = convolutionalNeuralNetwork(input_shape = input_shape, num_classes = num_classes)
    detector._build_model()

    start_time = time.time()
    training_time = time.time() - start_time
    print(f"Time to build and compile the model: {training_time:.2f} seconds")

    start_time = time.time()
    detector.train(X, y, X, y, epochs=30, batch_size=32)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    start_time = time.time()
    evaluation_results = detector.evaluate(X, y)
    eval_time = time.time() - start_time
    print(f"Evaluation time: {eval_time:.2f} seconds")

    print(f"Loss: {evaluation_results[0]}")
    print(f"Accuracy: {evaluation_results[1]}")

    canvas.delete("all")
    train_canvas = create_save_model_canvas(canvas, detector, label_encoder)
    train_canvas.pack(fill="both", expand=True)


def create_choose_file_train_canvas(canvas):
    create_my_text(canvas, 59, 103, "Plik z danymi", "w")
    create_button(canvas, 59.0, 147.0, 609.0, 222.0, "", "Lookup", "#FFFFFF")
    text_id = create_my_text(canvas, 64, 170, "", "nw")

    create_my_text(canvas, 168, 260, "Czy pierwszy wiersz to \n etykiety kolumn", "nw")
    _, check_id = create_button(canvas, 59.0, 265.0, 140.0, 340.0, "", "Checkbox", "#FFFFFF")

    create_button(canvas, 180.0, 411.0, 465.0, 486.0, "Przeglądaj", "Search")

    create_button(canvas, 180.0, 526.0, 465.0, 601.0, "Dalej", "Next")

    filename_var = StringVar()

    canvas.tag_bind("Search", "<Button-1>", lambda event: browse_file(canvas, text_id, filename_var))
    canvas.tag_bind("Checkbox", "<Button-1>", lambda event: change_state(canvas, check_id))
    canvas.tag_bind("Next", "<Button-1>", lambda event: switch_to_save_model_canvas(canvas, filename_var, variable))

    return canvas
