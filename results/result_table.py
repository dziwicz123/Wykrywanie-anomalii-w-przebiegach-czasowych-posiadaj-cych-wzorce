import tkinter as tk
from tkinter import ttk
from results.result_plot import create_plot_canvas

def switch_to_chart(canvas, series_number, series_data, table_canvas):
    print(f"Wyświetlanie wykresu dla serii danych {series_number}")

    table_canvas.destroy()

    create_plot_canvas(canvas, series_data, series_number)



def on_mousewheel(event, canvas):
    canvas.yview_scroll(-1*(event.delta//120), "units")


def create_result_table_canvas(canvas, X, predicted_labels, true_labels=None):
    # Create table_canvas as a child of the provided canvas
    table_canvas = tk.Canvas(
        canvas,
        bg="#ABE3DC",
        height=450,
        width=550,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )
    table_canvas.place(x=60, y=180)

    # Create a frame in the canvas
    frame = tk.Frame(table_canvas, bg="#FFFFFF")

    # Add scrollbar to the canvas
    scrollbar = ttk.Scrollbar(table_canvas, orient=tk.VERTICAL)
    scrollbar.place(x=572, y=180, height=450)  # Adjust x, y, height to fit the table_canvas

    # Place the frame in the canvas
    table_canvas.create_window((0, 0), window=frame, anchor='nw')

    # Connect frame's yview to scrollbar's set method
    frame.bind("<Configure>", lambda e: table_canvas.configure(scrollregion=table_canvas.bbox("all")))
    scrollbar.config(command=table_canvas.yview)

    table_canvas.bind_all("<MouseWheel>", lambda event: on_mousewheel(event, table_canvas))

    if true_labels is not None:
        columns = ["Numer serii danych", "Wykryta anomalia", "Przypisana anomalia", "Pokaż wykres"]
    else:
        columns = ["Numer serii danych", "Wykryta anomalia", "Pokaż wykres"]

    # Adding headers to the table
    for col, text in enumerate(columns):
        label = tk.Label(frame, text=text, borderwidth=2, relief='ridge', padx=15, pady=10)
        label.grid(row=0, column=col, sticky='nsew')

    # Adding data to the table
    for i in range(len(X)):
        series_number = i + 1
        detected_anomaly = predicted_labels[i]

        tk.Label(frame, text=series_number, borderwidth=2, relief='ridge', padx=10, pady=5).grid(row=i + 1, column=0,
                                                                                                 sticky='nsew')
        tk.Label(frame, text=detected_anomaly, borderwidth=2, relief='ridge', padx=10, pady=5).grid(row=i + 1, column=1,
                                                                                                    sticky='nsew')

        if true_labels is not None:
            assigned_anomaly = true_labels[i]
            tk.Label(frame, text=assigned_anomaly, borderwidth=2, relief='ridge', padx=10, pady=5).grid(row=i + 1,
                                                                                                        column=2,
                                                                                                        sticky='nsew')

            button = tk.Button(frame, text="Wyświetl", bg='green', fg='white',
                               command=lambda s=series_number: switch_to_chart(canvas, s, X[s - 1], table_canvas))
            button.grid(row=i + 1, column=3, sticky='nsew')
        else:
            button = tk.Button(frame, text="Wyświetl", bg='green', fg='white',
                               command=lambda s=series_number: switch_to_chart(canvas, s, X[s - 1], table_canvas))
            button.grid(row=i + 1, column=2, sticky='nsew')

    # Setting column proportions
    for col in range(len(columns)):
        frame.grid_columnconfigure(col, weight=1)

    # Setting scrollable region
    frame.update_idletasks()
    table_canvas.configure(scrollregion=table_canvas.bbox("all"))

    return table_canvas

def on_mousewheel(event, canvas):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


