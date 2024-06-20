from tkinter import Frame, Label, filedialog, IntVar, StringVar
from creator import create_button, create_my_text
from results.result_main import create_result_main_canvas

import convolutionalNeuralNetwork as cnn

import numpy as np
import pandas as pd

variable1 = 0 #Pierwszy etykiety
variable2 = 0 #Czy zawiera etykiety

def browse_file(canvas, text_id, filename_var):
    filename = filedialog.askopenfilename(
        title="Wybierz plik",
        filetypes=[("Pliki csv", "*.csv"), ("Wszystkie pliki", "*.*")],
    )
    if filename:
        print(f"Wybrano plik: {filename}")
        canvas.itemconfig(text_id, text=f"{filename}", font=("Arial", 14))
        filename_var.set(filename)

def change_state(canvas, text_id, pos):
    global variable1, variable2
    
    if(pos==1):
        if(variable1 == 0):
            canvas.itemconfig(text_id, text="X")
            variable1 = 1
        else:
            canvas.itemconfig(text_id, text="")
            variable1 = 0
    else:
        if(variable2 == 0):
            canvas.itemconfig(text_id, text="X")
            variable2 = 1
        else:
            canvas.itemconfig(text_id, text="")
            variable2 = 0

def switch_to_result_main_canvas(canvas, detector, model_dir, file_dir, is_labeled=None, is_labeled_in_first_col=None):
    file_path = file_dir.get()
    data = pd.read_csv(file_path)

    folder_path = model_dir.get()

    if is_labeled is not None and is_labeled_in_first_col is not None and is_labeled == 1:
        if is_labeled_in_first_col == 1:
            y = data.iloc[:, 0].values
            X = data.drop(data.columns[0], axis=1).values
        else:
            y = data.iloc[:, -1].values
            X = data.drop(data.columns[-1], axis=1).values

        label_mapping_path = f'{folder_path}/label_mapping.json'
        label_mapping = cnn.load_label_mapping(label_mapping_path)

        y = cnn.convert_labels_to_one_hot(y, label_mapping)

    else:
        X = data.values
        y = None

    X = np.expand_dims(X, axis=-1)

    inverse_label_mapping = {np.argmax(v): k for k, v in label_mapping.items()}

    if y is not None:
        predictions = detector.model.predict(X)
        predicted_labels_indices = np.argmax(predictions, axis=1)
        predicted_labels = [inverse_label_mapping[idx] for idx in predicted_labels_indices]
        true_labels_indices = np.argmax(y, axis=1)
        true_labels = [inverse_label_mapping[idx] for idx in true_labels_indices]

        canvas.delete("all")
        train_canvas = create_result_main_canvas(canvas, X, predicted_labels, true_labels=true_labels)
        train_canvas.pack(fill="both", expand=True)

    else:
        predictions = detector.model.predict(X)
        predicted_labels_indices = np.argmax(predictions, axis=1)
        predicted_labels = [inverse_label_mapping[idx] for idx in predicted_labels_indices]

        canvas.delete("all")
        train_canvas = create_result_main_canvas(canvas, X, predicted_labels)
        train_canvas.pack(fill="both", expand=True)


def create_choose_file_test_canvas(canvas, detector, model_dir):
    create_my_text(canvas, 59, 103, "Plik z danymi", "w")
    create_button(canvas, 59.0, 147.0, 609.0, 222.0, "", "Lookup", "#FFFFFF")
    text_id = create_my_text(canvas, 64, 170, "", "nw")

    create_my_text(canvas, 168, 240, "Czy pierwszy wiersz to\netykiety kolumn", "nw")
    _, check_id1 = create_button(canvas, 59.0, 240.0, 59+75, 240+75, "", "Checkbox1", "#FFFFFF")

    create_my_text(canvas, 168, 343, "Czy dane są\nskategoryzowane", "nw")
    _, check_id2 = create_button(canvas, 59.0, 343.0, 59+75, 343+75, "", "Checkbox2", "#FFFFFF")

    create_button(canvas, 180.0, 457.0, 180+283, 457+75, "Przeglądaj", "Search")

    create_button(canvas, 180.0, 570.0, 180+283, 570+75, "Dalej", "Next")

    filename_var = StringVar()

    canvas.tag_bind("Search", "<Button-1>", lambda event: browse_file(canvas, text_id, filename_var))

    canvas.tag_bind("Checkbox1", "<Button-1>", lambda event: change_state(canvas, check_id1, variable1))
    canvas.tag_bind("Checkbox2", "<Button-1>", lambda event: change_state(canvas, check_id2, variable2))

    canvas.tag_bind("Next", "<Button-1>", lambda event: switch_to_result_main_canvas(canvas, detector, model_dir, filename_var, is_labeled=variable2, is_labeled_in_first_col=variable1))

    return canvas
