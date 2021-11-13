"""
SOBEL OPERATOR OR SOBEL-FELDMAN OPERATOR
It is used in image processing and computer vision, particularly within edge detection algorithms where it creates an
image emphasising edges.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # triangle filter
    hx = np.array([[1, 2, 1]])
    # central change
    hy = np.array([[1, 0, -1]])
    # Sobel operator
    Hx = hx.T * hy
    Hy = hy.T * hx
    print(Hx)
    print(Hy)
    # Gx
    # Gy
