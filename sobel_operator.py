"""
SOBEL OPERATOR OR SOBEL-FELDMAN OPERATOR
It is used in image processing and computer vision, particularly within edge detection algorithms where it creates an
image emphasising edges.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

if __name__ == "__main__":
    # triangle filter
    hx = np.array([[1, 2, 1]])
    # central change
    hy = np.array([[1, 0, -1]])
    # Sobel operator
    Hx = hx.T * hy
    Hy = hy.T * hx

    # image reading
    list_path = os.listdir('C:/Users/any12/PycharmProjects/Computer_vision/sobel_operator_images/')
    for i in range(len(list_path)):
        image = cv2.imread(f"C:/Users/any12/PycharmProjects/Computer_vision/sobel_operator_images/{list_path[i]}", cv2.IMREAD_GRAYSCALE)
        plt.figure(list_path[i])
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title("Source image")
        ax1 = plt.imshow(image, cmap="gray")

        Gx = cv2.Sobel(image, -1, 1, 0, 3)
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title("Gx")
        ax2 = plt.imshow(Gx, cmap="gray")

        Gy = cv2.Sobel(image, -1, 0, 1, 3)
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title("Gy")
        ax3 = plt.imshow(Gy, cmap="gray")

        G = cv2.addWeighted(Gx, 0.5, Gy, 0.5, 0)
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title("G")
        ax4 = plt.imshow(G, cmap="gray")

    plt.show()
