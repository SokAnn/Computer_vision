'''
IMAGE OBJECT RECOGNITION IN THE LOG-POLAR COORDINATE SYSTEM AND DETERMINING THE PARAMETERS OF ITS ROTATION AND SCALING
Recognizing objects in an image and determining the parameters of their spatial rotation and scaling with respect to a
source image using a correlation method in a polar-logarithmic coordinate system.
'''

import cv2
import numpy as np
import os

def read_image(image, text):
    img = cv2.imread(image)
    cv2.namedWindow(f"{text}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{text}", img)
    return img

def log_polar(image, text):
    p = cv2.logPolar(image, (image.shape[0] / 2, image.shape[1] / 2), 40, cv2.WARP_FILL_OUTLIERS)
    cv2.namedWindow(f"{text} in log-polar coordinate system", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{text} in log-polar coordinate system", p)
    return p

def cross_correlation(image, text):
    t = np.concatenate((image, image), axis=0)
    small = np.concatenate((t, t), axis=1)
    corr = cv2.matchTemplate(image, small, cv2.TM_CCORR_NORMED)
    cv2.namedWindow(f"Cross-correlation function for {text}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation function for {text}", corr)

def scale_image(image, percent):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    scaled_image = (width, height)
    scaled_image = cv2.resize(image, scaled_image, interpolation=cv2.INTER_AREA)
    cv2.namedWindow(f"Image scaled by {percent}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Image scaled by {percent}", scaled_image)
    return scaled_image

if __name__ == '__main__':
    list_path = os.listdir('C:/Users/any12/PycharmProjects/Computer_vision/object_recognition_templates/')

    for i in range(len(list_path)):
        im = read_image('C:/Users/any12/PycharmProjects/Computer_vision/object_recognition_templates/' + list_path[i], list_path[i])
        lp = log_polar(im, list_path[i])
        cross_correlation(lp, list_path[i])

    original = cv2.imread('C:/Users/any12/PycharmProjects/Computer_vision/object_recognition_templates/' + list_path[0])

    for i in range(115, 145, 15):
        si = scale_image(original, i)
        lp = log_polar(si, f"{i}")
        cross_correlation(lp, f"{i}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

