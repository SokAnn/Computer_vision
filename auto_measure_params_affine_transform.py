"""
AUTOMATIC MEASUREMENT OF IMAGE AFFINE TRANSFORMATION PARAMETERS
The purpose of the work is to measure the parameters of the affine transformation of images and normalize them.
"""

import cv2
import numpy as np
import math as mt
from PIL import Image
import os

if __name__ == '__main__':
    list_path = os.listdir('C:/Users/any12/PycharmProjects/Computer_vision/auto_measure_params_affine_transform_images/')
    image = Image.open('C:/Users/any12/PycharmProjects/Computer_vision/auto_measure_params_affine_transform_images/' + list_path[0])
    image = image.convert('L')
    image = np.array(image)
    mas = image.copy()
    mas = np.reshape(mas, -1)

    for i in range(1, len(list_path)):
        temp_mas = mas.copy()
        temp_image = Image.open('C:/Users/any12/PycharmProjects/Computer_vision/auto_measure_params_affine_transform_images/' + list_path[i])
        size_im_x, size_im_y = temp_image.size
        temp_image = temp_image.convert('L')
        temp_image = np.array(temp_image)
        temp_image = np.reshape(temp_image, -1)

        x_t = 0
        y_t = 0
        t = 0
        B = 0
        C = 0
        D = 0
        # Вычисление абсциссы и ординаты центра тяжести изображения
        for ii in range(0, size_im_x):
            for j in range(0, size_im_y):
                x_t += j * temp_image[ii * size_im_x + j]
                t += temp_image[ii * size_im_x + j]
                y_t += ii * temp_image[ii * size_im_x + j]
        xc = x_t / t
        yc = y_t / t
        print(xc, yc)
        # Вычисление направления и величины сжатия изображения
        for ii in range(0, size_im_x):
            for j in range(0, size_im_y):
                B += temp_image[ii * size_im_x + j] * (((j - xc) * (j - xc)) - ((ii - yc) * (ii - yc)))
                C += temp_image[ii * size_im_x + j] * 2 * (j - xc) * (ii - yc)
                D += temp_image[ii * size_im_x + j] * (((j - xc) * (j - xc)) + ((ii - yc) * (ii - yc)))
        print(B, C, D)
        # Величина сжатия изображения
        mu = mt.sqrt((D + mt.sqrt((C * C) + (B * B))) / (D - mt.sqrt((C * C) + (B * B))))
        # Направление сжатия изображения
        teta = (0.5 * mt.atan2(C, B))
        print(mu, teta)

        summ1 = 0
        summ2 = 0
        # Расчёт абциссы и ординаты пикселов центрированного изображения после компенсации его масштабирования
        for ii in range(0, size_im_x):
            for j in range(0, size_im_y):
                x_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (ii - yc) * mt.sin(-teta)) * mt.cos(teta) - (
                            (j - xc) * mt.sin(-teta) + (ii - yc) * mt.cos(-teta)) * mt.sin(teta)
                y_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (ii - yc) * mt.sin(-teta)) * mt.sin(teta) + (
                            (j - xc) * mt.sin(-teta) + (ii - yc) * mt.cos(-teta)) * mt.cos(teta)
                summ1 += temp_image[ii * size_im_x + j] * mt.sqrt((x_pls * x_pls) + (y_pls * y_pls))
                summ2 += temp_image[ii * size_im_x + j]
        print(x_pls, y_pls)
        print(summ1)
        print(summ2)

        K = 10
        # Величина равномерного масштабирования изображения
        M = summ1 / (K * summ2)
        print(M)
        for ii in range(0, size_im_x):
            for j in range(0, size_im_y):
                x_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (ii - yc) * mt.sin(-teta)) * mt.cos(teta) - (
                            (j - xc) * mt.sin(-teta) + (ii - yc) * mt.cos(-teta)) * mt.sin(teta)
                y_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (ii - yc) * mt.sin(-teta)) * mt.sin(teta) + (
                            (j - xc) * mt.sin(-teta) + (ii - yc) * mt.cos(-teta)) * mt.cos(teta)
                x = (1 / M) * x_pls
                y = (1 / M) * y_pls
                xi = x + xc
                yi = y + yc
                x = int(x)
                y = int(y)
                xi = int(xi)
                yi = int(yi)
                # Центрируем в обратную сторону, чтобы изображение было в центре
                if (xi > 0) and (yi > 0) and (xi <= size_im_x) and (yi <= size_im_x):
                    temp_mas[yi * size_im_x + xi] = temp_image[ii * size_im_x + j]

        temp_image = temp_image.reshape(size_im_x, size_im_x)
        cv2.imshow(f"{list_path[i]}", temp_image)
        temp_mas = temp_mas.reshape(size_im_x, size_im_x)
        cv2.imshow(f"result for {list_path[i]}", temp_mas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
