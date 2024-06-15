import tkinter as tk
from creator import create_my_text, create_button
import main_menu
import convolutionalNeuralNetwork as cnn

def switch_to_main_menu_canvas(canvas):
    canvas.delete("all")
    train_canvas = main_menu.create_main_menu_frame(canvas)
    train_canvas.pack(fill="both", expand=True)

def save_model(event, entry, canvas, detector, label_encoder):
    model_name = entry.get()
    print(f"Nazwa modelu: {model_name}")

    detector.save_model(f'models/{model_name}/model')
    cnn.save_label_mapping(label_encoder, f'models/{model_name}/label_mapping.json')

    switch_to_main_menu_canvas(canvas)

def create_save_model_canvas(canvas, detector, label_encoder):
    create_my_text(canvas, 343, 145, "Model został pomyślnie \n wytrenowany")
    create_my_text(canvas, 59.0, 284.0, "Wprowadź nazwę modelu", "w")
    
    entry = tk.Entry(canvas, font=("Arial", 30), bd=0, highlightthickness=0, relief='flat', bg='#FFFFFF')
    canvas.create_window(59.0, 328.0, anchor="nw", window=entry, width=550, height=75)

    create_button(canvas, 136.0, 484.0, 536.0, 559.0, "Zapisz model", "Save_model")

    # Powiązanie przycisku "Zapisz model" z funkcją save_model z przekazaniem argumentów
    canvas.tag_bind("Save_model", "<Button-1>", lambda event: save_model(event, entry, canvas, detector, label_encoder))

    return canvas