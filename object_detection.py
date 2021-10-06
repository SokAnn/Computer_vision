'''
OBJECT DETECTION AND MEASURING ITS CARTESIAN COORDINATES IN IMAGE BY ANALYSIS OF CROSS-CORRELATION FUNCTION
Recognition of an object and determination of its Cartesian coordinates in the image by calculating the
cross-correlation function in the space-time domain.
'''

import cv2

def show_function(origin_image, template_1, template_2):
    # show origin
    image = cv2.imread(origin_image)
    cv2.namedWindow("Origin image", cv2.WINDOW_NORMAL)
    cv2.imshow("Origin image", image)
    # show template 1
    temp_1 = cv2.imread(template_1)
    cv2.namedWindow("Template image #1", cv2.WINDOW_NORMAL)
    cv2.imshow("Template image #1", temp_1)
    # show template 2
    temp_2 = cv2.imread(template_2)
    cv2.namedWindow("Template image #2", cv2.WINDOW_NORMAL)
    cv2.imshow("Template image #2", temp_2)
    return [image, temp_1, temp_2]

def cross_correlation(origin_image, template_1, template_2):
    # cross-correlation function for template 1
    cross_corr_1 = cv2.matchTemplate(origin_image, template_1, cv2.TM_CCORR_NORMED)# TM_CCORR_NORMED
    cross_corr_1 = cv2.normalize(cross_corr_1, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow("Cross-correlation function for template 1", cv2.WINDOW_NORMAL)
    cv2.imshow("Cross-correlation function for template 1", cross_corr_1)
    # cross-correlation function for template 2
    cross_corr_2 = cv2.matchTemplate(origin_image, template_2, cv2.TM_CCORR_NORMED)# TM_CCORR_NORMED
    cross_corr_2 = cv2.normalize(cross_corr_2, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow("Cross-correlation function for template 2", cv2.WINDOW_NORMAL)
    cv2.imshow("Cross-correlation function for template 2", cross_corr_2)

def cross_correlation_rotate(origin_image, rotated_image_1, rotated_image_2, rotated_image_3, text):
    # rotated template
    r1 = cv2.imread(rotated_image_1)
    cv2.namedWindow(f"Rotated {text} by 2", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Rotated {text} by 2", r1)
    # cross-correlation
    ccr1 = cv2.matchTemplate(origin_image, r1, cv2.TM_CCORR_NORMED)
    ccr1 = cv2.normalize(ccr1, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for rotated {text} by 2", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for rotated {text} by 2", ccr1)
    # rotated template
    r2 = cv2.imread(rotated_image_2)
    cv2.namedWindow(f"Rotated {text} by 5", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Rotated {text} by 5", r2)
    # cross-correlation
    ccr2 = cv2.matchTemplate(origin_image, r2, cv2.TM_CCORR_NORMED)
    ccr2 = cv2.normalize(ccr2, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for rotated {text} by 5", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for rotated {text} by 5", ccr2)
    # rotated template
    r3 = cv2.imread(rotated_image_3)
    cv2.namedWindow(f"Rotated {text} by 10", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Rotated {text} by 10", r3)
    # cross-correlation
    ccr3 = cv2.matchTemplate(origin_image, r3, cv2.TM_CCORR_NORMED)
    ccr3 = cv2.normalize(ccr3, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for rotated {text} by 10", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for rotated {text} by 10", ccr3)

def cross_correlation_scale(origin_image, scaled_image_1, scaled_image_2, scaled_image_3, scaled_image_4, text):
    # scaled template
    s1 = cv2.imread(scaled_image_1)
    cv2.namedWindow(f"Scaled {text} by 2", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Scaled {text} by 2", s1)
    # cross-correlation
    ccs1 = cv2.matchTemplate(origin_image, s1, cv2.TM_CCORR_NORMED)
    ccs1 = cv2.normalize(ccs1, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for scaled {text} by 2", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for scaled {text} by 2", ccs1)
    # scaled template
    s2 = cv2.imread(scaled_image_2)
    cv2.namedWindow(f"Scaled {text} by 5", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Scaled {text} by 5", s2)
    # cross-correlation
    ccs2 = cv2.matchTemplate(origin_image, s2, cv2.TM_CCORR_NORMED)
    ccs2 = cv2.normalize(ccs2, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for scaled {text} by 5", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for scaled {text} by 5", ccs2)
    # scaled template
    s3 = cv2.imread(scaled_image_3)
    cv2.namedWindow(f"Scaled {text} by 10", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Scaled {text} by 10", s3)
    # cross-correlation
    ccs3 = cv2.matchTemplate(origin_image, s3, cv2.TM_CCORR_NORMED)
    ccs3 = cv2.normalize(ccs3, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for scaled {text} by 10", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for scaled {text} by 10", ccs3)
    # scaled temlate
    s4 = cv2.imread(scaled_image_4)
    cv2.namedWindow(f"Scaled {text} by 20", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Scaled {text} by 20", s4)
    # cross-correlation
    ccs4 = cv2.matchTemplate(origin_image, s4, cv2.TM_CCORR_NORMED)
    ccs4 = cv2.normalize(ccs4, None, 1, 0, cv2.NORM_MINMAX)
    cv2.namedWindow(f"Cross-correlation for scaled {text} by 20", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Cross-correlation for scaled {text} by 20", ccs4)

if __name__ == '__main__':
    images = show_function('C:/Users/any12/PycharmProjects/Computer_vision/origin.jpg',
                           'C:/Users/any12/PycharmProjects/Computer_vision/template1.jpg',
                           'C:/Users/any12/PycharmProjects/Computer_vision/template2.jpg')

    cross_correlation(images[0], images[1], images[2])
    cross_correlation_rotate(images[0], 'C:/Users/any12/PycharmProjects/Computer_vision/rotated_template1_2.jpg',
                             'C:/Users/any12/PycharmProjects/Computer_vision/rotated_template1_5.jpg',
                             'C:/Users/any12/PycharmProjects/Computer_vision/rotated_template1_10.jpg', 'template #1')
    cross_correlation_scale(images[0], 'C:/Users/any12/PycharmProjects/Computer_vision/scaled_template1_2.jpg',
                            'C:/Users/any12/PycharmProjects/Computer_vision/scaled_template1_5.jpg',
                            'C:/Users/any12/PycharmProjects/Computer_vision/scaled_template1_10.jpg',
                            'C:/Users/any12/PycharmProjects/Computer_vision/scaled_template1_20.jpg', 'template #1')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
