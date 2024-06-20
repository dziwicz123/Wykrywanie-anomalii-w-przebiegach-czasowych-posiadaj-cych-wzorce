from creator import create_my_text, create_button
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def create_result_stats_canvas(canvas, predicted_labels, true_labels):
    canvas.create_line(340, 629-450, 340, 629, fill="black", width=2)
    canvas.create_line(50, 289, 50+550, 289, fill="black", width=2)
    canvas.create_line(50, 404, 50+550, 404, fill="black", width=2)
    canvas.create_line(50, 513, 50+550, 513, fill="black", width=2)

    accuracy = round(accuracy_score(true_labels, predicted_labels) * 100, 2)
    precision = round(precision_score(true_labels, predicted_labels, average='weighted') * 100, 2)
    recall = round(recall_score(true_labels, predicted_labels, average='weighted') * 100, 2)
    f1 = round(f1_score(true_labels, predicted_labels, average='weighted') * 100, 2)

    create_my_text(canvas, 99, 216, "Precision", "nw")
    create_my_text(canvas, 417, 216, f"{precision}%", "nw")

    create_my_text(canvas, 99, 324, "Accuracy", "nw")
    create_my_text(canvas, 417, 324, f"{accuracy}%", "nw")

    create_my_text(canvas, 99, 432, "F1-Score", "nw")
    create_my_text(canvas, 417, 432, f"{f1}%", "nw")

    create_my_text(canvas, 99, 540, "Recall", "nw")
    create_my_text(canvas, 417, 540, f"{recall}%", "nw")

    return canvas