

def create_my_text(canvas, x, y, text, anchor="center"):
    return canvas.create_text(x, y, text=text, fill="#000000", font=("Arial", 30, "bold"), anchor=anchor)

def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=50, **kwargs):
    if x2 - x1 < 2*radius:
        radius = (x2 - x1) / 2
    if y2 - y1 < 2*radius:
        radius = (y2 - y1) / 2

    points = [
        x1+radius, y1,
        x1+radius, y1,
        x2-radius, y1,
        x2-radius, y1,
        x2, y1,
        x2, y1+radius,
        x2, y1+radius,
        x2, y2-radius,
        x2, y2-radius,
        x2, y2,
        x2-radius, y2,
        x2-radius, y2,
        x1+radius, y2,
        x1+radius, y2,
        x1, y2,
        x1, y2-radius,
        x1, y2-radius,
        x1, y1+radius,
        x1, y1+radius,
        x1, y1,
    ]

    return canvas.create_polygon(points, smooth=True, **kwargs)

def create_button(canvas, x1, y1, x2, y2, text, button_name, color="#20C997"):
    # Utwórz prostokąt
    rect_id = create_rounded_rectangle(canvas, x1, y1, x2, y2, fill=color, outline="")

    # Oblicz środek prostokąta
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Dodaj wyśrodkowany tekst
    text_id = create_my_text(canvas, center_x, center_y, text)

    # Przypisz tagi do prostokąta i tekstu
    canvas.itemconfig(rect_id, tags=button_name)
    canvas.itemconfig(text_id, tags=button_name)

    return rect_id, text_id

