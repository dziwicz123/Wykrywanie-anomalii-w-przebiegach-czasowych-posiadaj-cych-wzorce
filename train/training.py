
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\Chronos\Desktop\build\assets\frame4")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("667x687")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 687,
    width = 667,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    667.0,
    687.0,
    fill="#ABE3DC",
    outline="")

canvas.create_rectangle(
    0.0,
    0.0,
    667.0,
    64.0,
    fill="#000000",
    outline="")

canvas.create_rectangle(
    59.0,
    244.0,
    609.0,
    363.0,
    fill="#000000",
    outline="")
window.resizable(False, False)
window.mainloop()
