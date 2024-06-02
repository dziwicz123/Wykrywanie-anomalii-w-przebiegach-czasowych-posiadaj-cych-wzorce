from creator import create_my_text, create_button
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

def create_plot_canvas(canvas, series_number=1):
    create_my_text(canvas, 60, 160, "Wybierz serię danych", "nw")

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x, y)
    ax.set_title(f"Wyświetlanie serii danych {series_number}")

    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    tk_image = ImageTk.PhotoImage(image=pil_image)

    canvas.create_image(60, 300, anchor='nw', image=tk_image)
    
    canvas.image = tk_image

    return canvas
