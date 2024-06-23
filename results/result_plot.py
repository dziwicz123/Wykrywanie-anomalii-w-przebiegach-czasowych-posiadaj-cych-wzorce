from creator import create_my_text, create_button
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk


def create_plot_canvas(canvas, series_data, series_number=1):
    #create_my_text(canvas, 60, 160, "Wybierz serię danych", "nw")

    # Generowanie osi X jako indeksy serii danych
    x = np.arange(len(series_data))
    y = series_data

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(x, y)
    ax.set_title(f"Wyświetlanie serii danych {series_number}")

    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    tk_image = ImageTk.PhotoImage(image=pil_image)

    canvas.create_image(60, 160, anchor='nw', image=tk_image)

    canvas.image = tk_image

    return canvas

