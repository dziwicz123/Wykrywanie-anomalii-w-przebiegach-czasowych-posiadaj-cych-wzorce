from tkinter import Frame, Label, filedialog, IntVar, StringVar
from creator import create_button, create_my_text
from testing.choose_file_test import create_choose_file_test_canvas
from convolutionalNeuralNetwork import convolutionalNeuralNetwork

def browse_file(canvas, text_id, folder_dir):
    foldername = filedialog.askdirectory(
        title="Wybierz folder modelu",
    )
    if foldername:
        print(f"Wybrano folder: {foldername}")
        canvas.itemconfig(text_id, text=f"{foldername}", font=("Arial", 14))
        folder_dir.set(foldername)


def switch_to_choose_file_test_canvas(canvas, folder_dir):
    detector = convolutionalNeuralNetwork(model_path=f'models/{folder_dir}/model')

    canvas.delete("all")
    test_canvas = create_choose_file_test_canvas(canvas, detector, folder_dir)
    test_canvas.pack(fill="both", expand=True)

def create_choose_model_canvas(canvas):
    create_my_text(canvas, 59, 103, "Wybierz model", "w")
    create_button(canvas, 59.0, 147.0, 609.0, 222.0, "", "Lookup", "#FFFFFF")
    text_id = create_my_text(canvas, 64, 170, "", "nw")

    create_button(canvas, 180.0, 457.0, 180+283, 457+75, "Przeglądaj", "Search")

    create_button(canvas, 180.0, 570.0, 180+283, 570+75, "Dalej", "Next")

    folder_dir = StringVar()
    
    canvas.tag_bind("Search", "<Button-1>", lambda event: browse_file(canvas, text_id, folder_dir))

    canvas.tag_bind("Next", "<Button-1>", lambda event: switch_to_choose_file_test_canvas(canvas, folder_dir))

    return canvas
