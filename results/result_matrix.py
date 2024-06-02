import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from creator import create_my_text

def plot_confusion_matrix(canvas, labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions)
    
    fig = Figure(figsize=(6, 6), dpi=100, facecolor="#ABE3DD")
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(cm, cmap=plt.cm.summer)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}\n({cm[i, j] / cm.sum() * 100:.1f}%)', 
                    ha='center', va='center',
                    color='black' if cm[i, j] > cm.max() / 2. else 'white')

    canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    canvas_agg.draw()
    window_id = canvas.create_window(70, 100, anchor='nw', window=canvas_agg.get_tk_widget())

    return window_id


def create_result_matrix_canvas(canvas):
    
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    predictions = [0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    class_names = ['0', '1']

    plot_confusion_matrix(canvas, labels, predictions, class_names)
    
    return canvas