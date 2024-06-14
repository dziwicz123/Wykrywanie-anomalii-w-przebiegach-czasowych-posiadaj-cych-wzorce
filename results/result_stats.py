from creator import create_my_text, create_button

def create_result_stats_canvas(canvas):
    canvas.create_line(340, 629-450, 340, 629, fill="black", width=2)
    canvas.create_line(50, 289, 50+550, 289, fill="black", width=2)
    canvas.create_line(50, 404, 50+550, 404, fill="black", width=2)
    canvas.create_line(50, 513, 50+550, 513, fill="black", width=2)

    create_my_text(canvas, 99, 216, "Precision", "nw")
    create_my_text(canvas, 417, 216, "100%", "nw")

    create_my_text(canvas, 99, 324, "Accuracy", "nw")
    create_my_text(canvas, 417, 324, "100%", "nw")

    create_my_text(canvas, 99, 432, "F1-Score", "nw")
    create_my_text(canvas, 417, 432, "100%", "nw")

    create_my_text(canvas, 99, 540, "Recall", "nw")
    create_my_text(canvas, 417, 540, "100%", "nw")

    return canvas