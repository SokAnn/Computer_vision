This repository presents implementations of some typical computer vision tasks, for example, image segmentation or image contour selection.
Each file solves a specific task:
- image_segm_otsu.py - threshold segmentation of the image by the Ocu method
- object_detection.py - object recognition and determination of its Cartesian coordinates in the image by calculating the mutual correlation function in the space-time domain
- object_recognition.py - recognition of the image object in the polar-logarithmic coordinate system with the definition of rotation and scaling parameters
- recognition_images_affine_transform.py - recognition of images subjected to affine transformations
- sobel_operator.py - Sobel operator for detecting image contours
- auto_measure_params_affine_transform.py - measurement of the parameters of the affine image transformation
- image_matching_FM_algorithm.py - image matching using the Fourier-Mellin algorithm

В данном репозитории представлены реализации некоторых типичных задач компьютерного зрения, например, сегментация изображения или выделение контуров изображения.
В каждом файле решается определенная задача:
- image_segm_otsu.py - пороговая сегментация изображения методом Оцу
- object_detection.py - распознавание объекта и определение его декартовых координат на изображении путем вычисления функции взаимной корреляции в пространственно-временной области
- object_recognition.py - распознавание объекта изображения в полярно-логарифмической системе координат с определением параметров поворота и масштабирования
- recognition_images_affine_transform.py - распознавание изображений, подвергнутым аффинным преобразованиям
- sobel_operator.py - оператор Собеля для выявления контуров изображения
- auto_measure_params_affine_transform.py - измерение параметров аффинного преобразования изображения
- image_matching_FM_algorithm.py - сопоставление изображений с использованием алгоритма Фурье-Меллина