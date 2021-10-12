'''
RECOGNITION OF IMAGES SUBJECTED TO AFFINE TRANSFORMS
'''

import cv2
import numpy as np
import math as mt
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # part 1
    print("part 1")
    for k in range(0, 4):
        print(k + 1, ' image processing...')
        if (k == 0):
            img1 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/image_1_7.jpg")
        if (k == 1):
            img1 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/image_2_7.jpg")
        if (k == 2):
            img1 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/image_3_7.jpg")
        if (k == 3):
            img1 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/image_4_7.jpg")

        imgf = img1.convert('L')
        im = np.asarray(imgf, dtype=np.uint8)
        # cv2.namedWindow(f"Image {k + 1}", cv2.WINDOW_NORMAL)
        # cv2.imshow(f"Image {k + 1}", im)

        img111 = np.array(imgf)
        imm = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/image.jpg")
        imm_2 = imm.convert('L')
        ImTrf = imm_2.copy()
        ImTrf_im = np.array(ImTrf)
        ImTrf_im = np.reshape(ImTrf_im, -1)
        XSize, YSize = img1.size
        mass1 = [XSize * YSize]

        summ_x_t = 0
        summ_t = 0
        summ_y_t = 0
        B = 0
        C = 0
        D = 0

        mass1 = img111
        mass1 = np.reshape(img111, -1)

        # Вычисление абсциссы и ординаты центра тяжести изображения
        for i in range(0, XSize):
            for j in range(0, YSize):
                summ_x_t += j * mass1[i * XSize + j]
                summ_t += mass1[i * XSize + j]
                summ_y_t += i * mass1[i * XSize + j]
        xc = summ_x_t / summ_t
        yc = summ_y_t / summ_t
        print("xc = ", xc, "yc = ", yc)

        # Вычисление направления и величины сжатия изображения
        for i in range(0, XSize):
            for j in range(0, YSize):
                B += mass1[i * XSize + j] * (((j - xc) * (j - xc)) - ((i - yc) * (i - yc)))
                C += mass1[i * XSize + j] * 2 * (j - xc) * (i - yc)
                D += mass1[i * XSize + j] * (((j - xc) * (j - xc)) + ((i - yc) * (i - yc)))
        print("B = ", B, "C = ", C, "D = ", D)

        # Величина сжатия изображения
        mu = mt.sqrt((D + mt.sqrt((C * C) + (B * B))) / (D - mt.sqrt((C * C) + (B * B))))

        # Направление сжатия изображения
        teta = (0.5 * mt.atan2(C, B))
        print("mu = ", mu, "teta = ", teta)

        x_pls = 0
        y_pls = 0
        summ1 = 0
        summ2 = 0

        # Расчёт абциссы и ординаты пикселов центрированного изображения после компенсации его масштабирования
        for i in range(0, YSize):
            for j in range(0, XSize):
                x_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (i - yc) * mt.sin(-teta)) * mt.cos(teta) - (
                        (j - xc) * mt.sin(-teta) + (i - yc) * mt.cos(-teta)) * mt.sin(teta)
                y_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (i - yc) * mt.sin(-teta)) * mt.sin(teta) + (
                        (j - xc) * mt.sin(-teta) + (i - yc) * mt.cos(-teta)) * mt.cos(teta)

                summ1 += mass1[i * XSize + j] * mt.sqrt((x_pls * x_pls) + (y_pls * y_pls))
                summ2 += mass1[i * XSize + j]

        K = 10
        # Величина равномерного масштабирования изображения
        M = summ1 / (K * summ2)
        print("M = ", M)
        for i in range(0, YSize):
            for j in range(0, XSize):
                x_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (i - yc) * mt.sin(-teta)) * mt.cos(teta) - (
                        (j - xc) * mt.sin(-teta) + (i - yc) * mt.cos(-teta)) * mt.sin(teta)
                y_pls = (1 / mu) * ((j - xc) * mt.cos(-teta) - (i - yc) * mt.sin(-teta)) * mt.sin(teta) + (
                        (j - xc) * mt.sin(-teta) + (i - yc) * mt.cos(-teta)) * mt.cos(teta)

                x = (1 / M) * x_pls
                y = (1 / M) * y_pls
                xi = x + xc
                yi = y + yc
                x = int(x)
                y = int(y)
                xi = int(x + XSize / 2)
                yi = int(y + YSize / 2)
                # Центрируем в обратную сторону, чтобы изображ было в центре
                if (xi > 0) and (yi > 0) and (xi <= XSize) and (yi <= YSize):
                    ImTrf_im[yi * XSize + xi] = mass1[i * XSize + j]

        ImTrf_im = ImTrf_im.reshape(XSize, YSize)

        # cv2.namedWindow(f"Image {k + 1} (result)", cv2.WINDOW_NORMAL)
        # cv2.imshow(f"Image {k + 1} (result)", ImTrf_im)

        ImTrf_im = cv2.resize(ImTrf_im, (0, 0), fx=5, fy=5)
        cv2.imwrite(f"C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/Results/RESULT{k + 1}.jpg", ImTrf_im)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # part 2
    print("part 2")
    # число секторов
    N = 256
    for i in range(0, 4):
        if i == 0:
            img1 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/Results/RESULT1.jpg")
            img1 = img1.convert('L')
            img1 = np.asarray(img1, dtype='d')
        if i == 1:
            img2 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/Results/RESULT2.jpg")
            img2 = img2.convert('L')
            img2 = np.asarray(img2, dtype='d')
        if i == 2:
            img3 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/Results/RESULT3.jpg")
            img3 = img3.convert('L')
            img3 = np.asarray(img3, dtype='d')
        if i == 3:
            img4 = Image.open("C:/Users/any12/PycharmProjects/Computer_vision/recognition_images_affine_transforms/Results/RESULT4.jpg")
            img4 = img4.convert('L')
            img4 = np.asarray(img4, dtype='d')

    # определение азимутальной координаты
    YSize = XSize = len(img1)
    coord = np.zeros((XSize, YSize))
    for i in range(0, XSize):
        for j in range(0, YSize):
            coord[i][j] = mt.atan2(i - XSize / 2, j - YSize / 2)

    # сектора
    diap = np.arange(0, 2 * np.pi, 2 * np.pi/N)

    # подсчет суммы пикселов всего изображения
    sum_all_1 = 0
    sum_all_2 = 0
    sum_all_3 = 0
    sum_all_4 = 0
    for i in range(0, XSize):
        for j in range(0, YSize):
            sum_all_1 = sum_all_1 + img1[i][j]
            sum_all_2 = sum_all_2 + img2[i][j]
            sum_all_3 = sum_all_3 + img3[i][j]
            sum_all_4 = sum_all_4 + img4[i][j]

    # подсчет суммы для каждого сектора
    sum_sector_1 = np.zeros((1, N))
    sum_sector_2 = np.zeros((1, N))
    sum_sector_3 = np.zeros((1, N))
    sum_sector_4 = np.zeros((1, N))
    for i in range(0, XSize):
        for j in range(0, YSize):
            for n in range(0, N):
                if coord[i][j] >= diap[n] and coord[i][j] < diap[n + 1]:
                    sum_sector_1[0][n] = sum_sector_1[0][n] + img1[i][j]
                    sum_sector_2[0][n] = sum_sector_2[0][n] + img2[i][j]
                    sum_sector_3[0][n] = sum_sector_3[0][n] + img3[i][j]
                    sum_sector_4[0][n] = sum_sector_4[0][n] + img4[i][j]
    sum_sector_1 = sum_sector_1 / sum_all_1
    sum_sector_2 = sum_sector_2 / sum_all_2
    sum_sector_3 = sum_sector_3 / sum_all_3
    sum_sector_4 = sum_sector_4 / sum_all_4

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(sum_sector_1[0])
    ax2.plot(sum_sector_2[0])
    ax3.plot(sum_sector_3[0])
    ax4.plot(sum_sector_4[0])
    plt.show()

    f1 = np.fft.fft2(sum_sector_1)
    f2 = np.fft.fft2(sum_sector_2)
    f3 = np.fft.fft2(sum_sector_3)
    f4 = np.fft.fft2(sum_sector_4)

    af1 = abs(f1)
    af2 = abs(f2)
    af3 = abs(f3)
    af4 = abs(f4)

    # попарное расстояние, Евклидово расстояние, Евклидова норма (по умолчанию)
    distance11 = np.linalg.norm(af1 - af1)
    distance21 = np.linalg.norm(af2 - af1)
    distance31 = np.linalg.norm(af3 - af1)
    distance41 = np.linalg.norm(af4 - af1)

    graph = [distance11, distance21, distance31, distance41]
    plt.bar([1, 2, 3, 4], graph)
    plt.xlabel("Number of image")
    plt.ylabel("The result of comparing images")
    plt.show()
