from tkinter import Frame, Label, filedialog, IntVar
from creator import create_button, create_my_text
from results.result_main import create_result_main_canvas

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

def switch_to_result_main_canvas(canvas):
    canvas.delete("all")
    train_canvas = create_result_main_canvas(canvas)
    train_canvas.pack(fill="both", expand=True)

def create_choose_file_test_canvas(canvas):
    create_my_text(canvas, 59, 103, "Plik z danymi", "w")
    create_button(canvas, 59.0, 147.0, 609.0, 222.0, "", "Lookup", "#FFFFFF")
    text_id = create_my_text(canvas, 64, 170, "", "nw")

    create_my_text(canvas, 168, 240, "Czy pierwszy wiersz to \netykiety kolumn", "nw")
    _, check_id1 = create_button(canvas, 59.0, 240.0, 59+75, 240+75, "", "Checkbox1", "#FFFFFF")

    create_my_text(canvas, 168, 343, "Czy dane są \nskategoryzowane", "nw")
    _, check_id2 = create_button(canvas, 59.0, 343.0, 59+75, 343+75, "", "Checkbox2", "#FFFFFF")

    create_button(canvas, 180.0, 457.0, 180+283, 457+75, "Przeglądaj", "Search")

    create_button(canvas, 180.0, 570.0, 180+283, 570+75, "Dalej", "Next")
    
    canvas.tag_bind("Search", "<Button-1>", lambda event: browse_file(canvas, text_id))

    variable1 = IntVar()
    variable2 = IntVar()

    canvas.tag_bind("Checkbox1", "<Button-1>", lambda event: change_state(canvas, check_id1, variable1))
    canvas.tag_bind("Checkbox2", "<Button-1>", lambda event: change_state(canvas, check_id2, variable2))

    canvas.tag_bind("Next", "<Button-1>", lambda event: switch_to_result_main_canvas(canvas))

    return canvas
