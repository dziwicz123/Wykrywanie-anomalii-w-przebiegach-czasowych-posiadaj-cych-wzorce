from tkinter import Tk, Canvas, Button
from creator import create_my_text, create_rounded_rectangle
from main_menu import create_main_menu_frame
from train.choose_file_train import *

def on_drag(event):
    window.geometry(f"+{event.x_root - x}+{event.y_root - y}")

def on_press(event):
    global x, y
    x, y = event.x, event.y

def minimize_root_drawn():
    window.state('withdrawn')
    window.overrideredirect(False)
    window.state('iconic')

def minimize_root_threw(a):
    window.overrideredirect(True)

def close_window():
    root = canvas.winfo_toplevel()
    canvas.delete("all")
    root.quit()

def create_top_bar(canvas):
    create_rounded_rectangle(canvas, 0, 0, 667, 64, fill="#2C6DD2", outline="")
    canvas.create_rectangle(0, 0, 667, 30, fill="#2C6DD2", outline="")

    create_my_text(canvas, 200, 32, text="Anomaly Detection")

    minimize_button = Button(canvas, text="_", command=minimize_root_drawn, bg="#2C6DD2", fg="black", bd=0, padx=0, pady=0, font=("Arial", 20, "bold"))
    minimize_button.place(x=580, y=5)

    close_button = Button(canvas, text="X", command=close_window, bg="#2C6DD2", fg="black", bd=0, padx=2, pady=0, font=("Arial", 20, "bold"))
    close_button.place(x=620, y=5)

def on_click(icon):
    window.deiconify()

window = Tk()
window.geometry("667x687")

canvas = Canvas(
    window,
    bg="#ABE3DC",
    height=687,
    width=667,
    bd=0,
    highlightthickness=0,
    relief="ridge",
)
canvas.place(x=0, y=0)

top_bar_canvas = Canvas(
    window,
    bg="#ABE3DC",
    height=64,
    width=667,
    bd=0,
    highlightthickness=0,
    relief="ridge",
)
top_bar_canvas.place(x=0, y=0)

create_top_bar(top_bar_canvas)

x, y = 0, 0
top_bar_canvas.bind("<ButtonPress-1>", on_press)
top_bar_canvas.bind("<B1-Motion>", on_drag)
window.bind("<Map>", minimize_root_threw)

main_menu_canvas = create_main_menu_frame(canvas)
main_menu_canvas.pack(side="bottom", fill="both", expand=True)

window.resizable(False, False)
window.mainloop()
