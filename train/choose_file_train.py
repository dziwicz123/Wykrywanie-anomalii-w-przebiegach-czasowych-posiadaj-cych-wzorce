from tkinter import Frame, Label, filedialog, IntVar
from creator import create_button, create_my_text
from train.save_model import create_save_model_canvas


def browse_file(canvas, text_id):
    filename = filedialog.askopenfilename(
        title="Wybierz plik",
        filetypes=[("Pliki csv", "*.csv"), ("Wszystkie pliki", "*.*")],
    )
    if filename:
        print(f"Wybrano plik: {filename}")
        canvas.itemconfig(text_id, text=f"{filename}", font=("Arial", 14))

def change_state(canvas, text_id, variable):
    if(variable.get() == 0):
        canvas.itemconfig(text_id, text="X")
        variable.set(1)
    else:
        canvas.itemconfig(text_id, text="")
        variable.set(0)

def switch_to_save_model_canvas(canvas):
    canvas.delete("all")
    train_canvas = create_save_model_canvas(canvas)
    train_canvas.pack(fill="both", expand=True)

def create_choose_file_train_canvas(canvas):
    create_my_text(canvas, 59, 103, "Plik z danymi", "w")
    create_button(canvas, 59.0, 147.0, 609.0, 222.0, "", "Lookup", "#FFFFFF")
    text_id = create_my_text(canvas, 64, 170, "", "nw")

    create_my_text(canvas, 168, 260, "Czy pierwszy wiersz to \n etykiety kolumn", "nw")
    _, check_id = create_button(canvas, 59.0, 265.0, 140.0, 340.0, "", "Checkbox", "#FFFFFF")

    create_button(canvas, 180.0, 411.0, 465.0, 486.0, "Przeglądaj", "Search")

    create_button(canvas, 180.0, 526.0, 465.0, 601.0, "Dalej", "Next")
    
    canvas.tag_bind("Search", "<Button-1>", lambda event: browse_file(canvas, text_id))

    variable = IntVar()

    canvas.tag_bind("Checkbox", "<Button-1>", lambda event: change_state(canvas, check_id, variable))

    canvas.tag_bind("Next", "<Button-1>", lambda event: switch_to_save_model_canvas(canvas))

    return canvas
