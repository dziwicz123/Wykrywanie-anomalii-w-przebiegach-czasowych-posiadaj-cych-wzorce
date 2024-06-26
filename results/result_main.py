from tkinter import Frame, Label, filedialog, IntVar, Canvas
from creator import create_button, create_my_text
from results.result_stats import create_result_stats_canvas
from results.result_matrix import create_result_matrix_canvas
from results.result_table import create_result_table_canvas
from results.result_plot import create_plot_canvas

table_canvas = None

def show_table(canvas, X, predicted_labels, true_labels=None):
    global table_canvas

    canvas.delete("all")
    if true_labels is not None:
        table_canvas = create_result_table_canvas(canvas, X, predicted_labels, true_labels=true_labels)
    else:
        table_canvas = create_result_table_canvas(canvas, X, predicted_labels)

def show_plot(canvas, X):
    global table_canvas
    if table_canvas is not None: table_canvas.destroy()

    canvas.delete("all")
    create_plot_canvas(canvas, X[0])

def show_matrix(canvas, predicted_labels, true_labels):
    global table_canvas
    if table_canvas is not None: table_canvas.destroy()

    canvas.delete("all")
    create_result_matrix_canvas(canvas, predicted_labels, true_labels)

def show_stats(canvas, predicted_labels, true_labels):
    global table_canvas
    if table_canvas is not None: table_canvas.destroy()

    canvas.delete("all")
    create_result_stats_canvas(canvas, predicted_labels, true_labels)


def create_result_bar_long(result_bar_canvas, canvas, X, predicted_labels, true_labels):
    create_button(result_bar_canvas, 0.0, 0.0, 167.0, 75.0, "Tabela", "Table")
    create_button(result_bar_canvas, 167.0, 0.0, 334.0, 75.0, "Wykres", "Chart")
    create_button(result_bar_canvas, 334.0, 0.0, 499.0, 75.0, "Macierz", "Matrix")
    create_button(result_bar_canvas, 499.0, 0.0, 499+168.0, 75.0, "Metryki", "Stats")

    result_bar_canvas.tag_bind("Table", "<Button-1>", lambda event: show_table(canvas, X, predicted_labels, true_labels))
    result_bar_canvas.tag_bind("Chart", "<Button-1>", lambda event: show_plot(canvas, X))
    result_bar_canvas.tag_bind("Matrix", "<Button-1>", lambda event: show_matrix(canvas, predicted_labels, true_labels))
    result_bar_canvas.tag_bind("Stats", "<Button-1>", lambda event: show_stats(canvas, predicted_labels, true_labels))

def create_result_bar_short(result_bar_canvas, canvas, X, predicted_labels):
    create_button(result_bar_canvas, 0.0, 0.0, 334.0, 75.0, "Tabela", "Table")
    create_button(result_bar_canvas, 334.0, 0.0, 499+168.0, 75.0, "Wykres", "Chart")

    result_bar_canvas.tag_bind("Table", "<Button-1>", lambda event: show_table(canvas, X, predicted_labels))
    result_bar_canvas.tag_bind("Chart", "<Button-1>", lambda event: show_plot(canvas, X))

def create_result_main_canvas(canvas, X, predicted_labels, true_labels=None):
    root = canvas.winfo_toplevel()

    result_bar_canvas = Canvas(
        root,
        bg="#ABE3DC",
        height=75,
        width=667,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )
    result_bar_canvas.place(x=0, y=64)

    if(true_labels is not None):
        create_result_bar_long(result_bar_canvas, canvas, X, predicted_labels, true_labels)
        show_table(canvas, X, predicted_labels, true_labels=true_labels)
    else:
        create_result_bar_short(result_bar_canvas, canvas, X, predicted_labels)
        show_table(canvas, X, predicted_labels, true_labels=None)

    return canvas
