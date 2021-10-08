'''
OBJECT DETECTION AND MEASURING ITS CARTESIAN COORDINATES IN IMAGE BY ANALYSIS OF CROSS-CORRELATION FUNCTION
Recognition of an object and determination of its Cartesian coordinates in the image by calculating the
cross-correlation function in the space-time domain.
'''

import cv2
import os

def read_and_show(image, text):
    # read & show
    image = cv2.imread(image)
    cv2.namedWindow(f"{text}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{text}", image)
    return image

def cross_correlation(origin_image, template, text):
    # cross-correlation function for template
    cross_corr = cv2.matchTemplate(origin_image, template, cv2.TM_CCORR_NORMED)
    cross_corr = cv2.normalize(cross_corr, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation function for {text}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation function for {text}", cross_corr)

if __name__ == '__main__':
    list_path = os.listdir('C:/Users/any12/PycharmProjects/Computer_vision/object_detection_templates/')
    images = []
    list_images = []

    for i in range(len(list_path)):
        # show origin image & templates
        images.append(read_and_show('C:/Users/any12/PycharmProjects/Computer_vision/object_detection_templates/' + list_path[i], list_path[i]))
        if not ('_' in list_path[i]):
            list_images.append(list_path[i])

    for i in range(len(list_path)):
        # computing cross-correlation function & show
        if '_' in list_path[i] and not(list_path[i] in list_images):
            cross_correlation(images[0], images[i], list_path[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
