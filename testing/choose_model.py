from tkinter import Frame, Label, filedialog, IntVar
from creator import create_button, create_my_text
from testing.choose_file_test import create_choose_file_test_canvas


def browse_file(canvas, text_id):
    filename = filedialog.askopenfilename(
        title="Wybierz plik",
        filetypes=[("Pliki csv", "*.csv"), ("Wszystkie pliki", "*.*")],
    )
    if filename:
        print(f"Wybrano plik: {filename}")
        canvas.itemconfig(text_id, text=f"{filename}", font=("Arial", 14))

def switch_to_choose_file_test_canvas(canvas):
    canvas.delete("all")
    test_canvas = create_choose_file_test_canvas(canvas)
    test_canvas.pack(fill="both", expand=True)

def create_choose_model_canvas(canvas):
    create_my_text(canvas, 59, 103, "Wybierz model", "w")
    create_button(canvas, 59.0, 147.0, 609.0, 222.0, "", "Lookup", "#FFFFFF")
    text_id = create_my_text(canvas, 64, 170, "", "nw")

    create_button(canvas, 180.0, 457.0, 180+283, 457+75, "Przeglądaj", "Search")

    create_button(canvas, 180.0, 570.0, 180+283, 570+75, "Dalej", "Next")
    
    canvas.tag_bind("Search", "<Button-1>", lambda event: browse_file(canvas, text_id))

    canvas.tag_bind("Next", "<Button-1>", lambda event: switch_to_choose_file_test_canvas(canvas))

    return canvas
