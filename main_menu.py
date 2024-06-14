from tkinter import Frame, PhotoImage, Label
from creator import create_button
from train.choose_file_train import create_choose_file_train_canvas
from testing.choose_model import create_choose_model_canvas

def switch_to_train_canvas(canvas):
    canvas.delete("all")
    train_canvas = create_choose_file_train_canvas(canvas)
    train_canvas.pack(fill="both", expand=True)

def switch_to_test_canvas(canvas):
    canvas.delete("all")
    train_canvas = create_choose_model_canvas(canvas)
    train_canvas.pack(fill="both", expand=True)

def close_window(canvas):
    root = canvas.winfo_toplevel()
    canvas.delete("all")
    root.destroy()

def create_main_menu_frame(canvas):

    create_button(canvas, 137.0, 372.0, 537.0, 447.0, "Trenowanie modelu", "Go_To_Train")
    create_button(canvas, 137.0, 474.0, 537.0, 549.0, "Testowanie modelu", "Go_To_Test")
    create_button(canvas, 134, 576, 533, 651, "Zamknij", "Exit")
    
    canvas.tag_bind("Go_To_Train", "<Button-1>", lambda event: switch_to_train_canvas(canvas))
    canvas.tag_bind("Go_To_Test", "<Button-1>", lambda event: switch_to_test_canvas(canvas))
    canvas.tag_bind("Exit", "<Button-1>", lambda event: close_window(canvas))

    image = PhotoImage(file="assets/logo.png")
    desired_width = 233
    image = image.subsample(round(image.width() / desired_width))

    canvas.image = image

    canvas.create_image(217, 98, image=image, anchor='nw')

    return canvas
