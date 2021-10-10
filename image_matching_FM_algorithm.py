'''
IMAGE MATCHING USING THE FOURIER-MELLIN ALGORITHM
The goal is to compute the correlation coefficient of the test image with each of the reference. To evaluate how the
correlation coefficient changes with the mutual scaling of the compared images, with mutual overlap and if the reference
image corresponds to the studied sample or does not correspond to it.
'''

import cv2
import numpy as np

RESULT = 1
def dft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def to_log_polar_coords(image):
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    w, h = image.shape[::-1]
    centre = (w / 2, h / 2)
    max_radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    log_polar_img = cv2.warpPolar(image, (w, h), centre, max_radius, flags)
    return log_polar_img

def zoom_image(image, percents):
    w, h = image.shape[::-1]
    centre = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(centre, 0, (percents / 100))
    transformed_image = cv2.warpAffine(image, matrix, (w, h))
    return transformed_image

def find_correlation(image, template_image):
    global RESULT
    m_i = dft(image)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/results/DFT_MAIN" + str(RESULT) + ".jpg", m_i)
    w, h = image.shape[::-1]
    t_i = dft(template_image)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/results/DFT_TEMPLATE" + str(RESULT) + ".jpg", t_i)
    log_polar_main = to_log_polar_coords(m_i)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/results/log_polar_main_img" + str(RESULT) + ".jpg", log_polar_main)
    log_polar_templ = to_log_polar_coords(t_i)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/results/log_polar_template_img" + str(RESULT) + ".jpg", log_polar_templ)
    m_i = dft(log_polar_main)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/results/main_log_polar_dft" + str(RESULT) + ".jpg", m_i)
    t_i = dft(log_polar_templ)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/results/template_log_polar_dft" + str(RESULT) + ".jpg", t_i)
    avg_B = np.mean(m_i)
    avg_B_m = np.mean(t_i)
    sum_num = 0
    sum_den_B = 0
    sum_den_B_m = 0
    for i in range(h):
        for j in range(w):
            sum_num += (m_i[i][j] - avg_B) * (t_i[i][j] - avg_B_m)
            sum_den_B += (m_i[i][j] - avg_B) ** 2
            sum_den_B_m += (t_i[i][j] - avg_B_m) ** 2
    corr_coeff = sum_num / np.sqrt(sum_den_B * sum_den_B_m)
    print("\t Correlation coefficient for test number " + str(RESULT) + ": ", str('{:.3f}'.format(corr_coeff)))
    RESULT += 1

if __name__ == "__main__":
    image = cv2.imread("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/invest_image.jpg", cv2.IMREAD_GRAYSCALE)
    # zoom image: 20, 50, 100 %
    im_zoom_20 = zoom_image(image, 120)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/zoom_20.jpg", im_zoom_20)
    im_zoom_50 = zoom_image(image, 150)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/zoom_50.jpg", im_zoom_50)
    im_zoom_100 = zoom_image(image, 200)
    cv2.imwrite("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/zoom_100.jpg", im_zoom_100)
    # shift image: 20, 40, 60 %
    shift_20 = cv2.imread("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/shift_20.jpg", cv2.IMREAD_GRAYSCALE)
    shift_40 = cv2.imread("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/shift_40.jpg", cv2.IMREAD_GRAYSCALE)
    shift_60 = cv2.imread("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/shift_60.jpg", cv2.IMREAD_GRAYSCALE)
    # foreign image
    foreign = cv2.imread("C:/Users/any12/PycharmProjects/Computer_vision/image_matching_FM_algorithm_images/foreign.jpg", cv2.IMREAD_GRAYSCALE)

    list_of_imgs = [image, im_zoom_20, im_zoom_50, im_zoom_100, shift_20, shift_40, shift_60, foreign]
    list(map(lambda img: find_correlation(image, img), list_of_imgs))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
