import matplotlib.pyplot as plt
import numpy as np


def show_marks(marks):
    
    polygons = []
    colors = []
    
    if len(marks) == 0:
        return

    # Sort marks by area in pixels
    sorted_marks = sorted(marks, key=(lambda m: m['area']), reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    for mark in sorted_marks:
        m = mark['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mark = np.random.random((1, 3)).tolist()[0]
        for i in range(1, 3):
            img[:, :, i] = color_mark[i]

        ax.imshow(np.dstack((img, m*0.35)))