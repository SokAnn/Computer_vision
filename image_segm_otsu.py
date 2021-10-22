"""
IMAGE SEGMENTATION BY THE OTSU THRESHOLDING METHOD
Segmentation images by the Otsu thresholding method.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu

if __name__ == "__main__":
    list_path = os.listdir('C:/Users/any12/PycharmProjects/Computer_vision/image_segmentation_otsu_method/')
    for i in range(len(list_path)):
        print(f"image {i + 1} processing...")
        image = cv2.imread(f"C:/Users/any12/PycharmProjects/Computer_vision/image_segmentation_otsu_method/{list_path[i]}", cv2.IMREAD_GRAYSCALE)
        for m in range(2, 6):
            thresholds = threshold_multiotsu(image, classes=m)
            print(thresholds)
            regions = np.digitize(image, bins=thresholds)

            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(image, cmap='gray')
            ax[0].set_title(f"Original image {i + 1}")
            ax[0].axis('off')

            ax[1].hist(image.ravel(), bins=255, color='b')
            ax[1].set_title(f"Histogram of image {i + 1}")
            for thresh in thresholds:
                ax[1].axvline(thresh, color='m')

            ax[2].imshow(regions, cmap='jet')
            ax[2].set_title(f"Segmentation of image {i + 1} by Otsu method (M = {m})")
            ax[2].axis('off')

        plt.show()
