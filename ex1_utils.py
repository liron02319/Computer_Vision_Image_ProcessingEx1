"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
import cv2
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 312324247


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    # 4.1

    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to GRAY_SCALE
    else:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

    normalize_image = image / 255.0   # normalize

    return normalize_image



def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    # 4.2

    img = imReadAndConvert(filename, representation)

    if representation == LOAD_GRAY_SCALE:
        cv2.imshow("GrayScaleImage", img)    # display as a GRAY_SCALE
        cv2.waitKey(0)
    else:
        plt.imshow(img)
        plt.show()      # display as a RGB



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    # 4.3A
    matrix = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])

    matrix_RGB2YIQ = np.dot(imgRGB, matrix.T.copy())   # Multiplication between matrices

    return matrix_RGB2YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
      Converts an YIQ image to RGB color space
      :param imgYIQ: An Image in YIQ
      :return: A RGB in image color space
      """

    # 4.3B
    matrix = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    matrix_inverse = np.linalg.inv(matrix)   # find the inverse matrix
    matrix_YIQ2RGB = np.dot(imgYIQ, matrix_inverse.T.copy())   # Multiplication between matrices

    return matrix_YIQ2RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # 4.4
    imageIsRGB = False  # flag check

    if len(imgOrig.shape) == 3:  # check that is RGB image
        imageIsRGB = True
        yiqImage = transformRGB2YIQ(imgOrig)   # transform image from RGB TO YIQ
        imgOrig = yiqImage[:, :, 0]    # Y channel of the YIQ image

    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')  # cast all the pixels to integers type
    histOrg = np.histogram(imgOrig.flatten(), bins=256)[0]  # Calculate Image Histogram
    cs = np.cumsum(histOrg)  # Calculate Cumulative Sum(CumSum)
    imgEq = cs[imgOrig]     # the new image with the equalized histogram
    imgEq = cv2.normalize(imgEq, None, 0, 255, cv2.NORM_MINMAX)
    imgEq = imgEq.astype('uint8') # cast all the pixels to integers type
    histEq = np.histogram(imgEq.flatten(), bins=256)[0]    # Calculate Image Histogram

    if imageIsRGB:  # flag check
        yiqImage[:, :, 0] = imgEq / 255  # Y channel of the YIQ image and normalize
        imgEq = transformYIQ2RGB(yiqImage)     # transform image from YIQ TO RGB

    return imgEq, histOrg, histEq



def caclculate_middle_boundary(_Z: np.ndarray, _Q: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
    """
        function that calculate between two boundary the middle means
        :param _Z:  _Z is an array that will represents the boundaries
        :param _Q:  _Q is an array that represent the values of the boundaries
        :return: (List[np.ndarray],List[np.ndarray])
    """

    # help function for quantizeImage

    for b in range(1, len(_Z) - 1):  # b is boundary
        _Z[b] = (_Q[b - 1] + _Q[b]) / 2  # function  that calculate between two boundary the middle means
    return _Z, _Q


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to *nQuant* colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # 4.5

    imageIsRGB = bool(imOrig.ndim == 3)  # check flag
    if imageIsRGB:  # RGB
        imageYIQ = transformRGB2YIQ(imOrig)   # transform image from RGB TO YIQ
        imOrig = np.copy(imageYIQ[:, :, 0])  # Y channel of the YIQ image
    else:  # Gray scale
        imageYIQ = imOrig

    if np.amax(imOrig) <= 1:  # check if imOrig is normalized
        imOrig = imOrig * 255

    imOrig = imOrig.astype('uint8')   # cast all the pixels to integers type
    histORGN = np.histogram(imOrig.flatten(), bins=256)[0]  # Calculate Image Histogram of origin image
    # find the boundaries
    size = int(255 / nQuant)  # divide the intervals evenly
    _Z = np.zeros(nQuant + 1, dtype=int)  # _Z is an array that will represents the boundaries

    for i in range(1, nQuant):   # run on the Number of colors to quantize the image
        _Z[i] = _Z[i - 1] + size  # boundaries coordinates of the image

    _Z[nQuant] = 255  # the left border will always start at 0 and the right border will always end at 255
    _Q = np.zeros(nQuant)  # _Q is an array that represent the values of the boundaries

    list_quantized = list()
    list_MSE = list()

    for i in range(nIter):  # run on the Number of optimization loops
        _newImg = np.zeros(imOrig.shape)  # Initialize a matrix with 0 in the original image size

        for j in range(len(_Q)):  # every j is a cell
            if j == len(_Q) - 1:  # last iterate of j
                right_cell = _Z[j + 1] + 1
            else:
                right_cell = _Z[j + 1]
            range_cell = np.arange(_Z[j], right_cell) #range from border to border
            _Q[j] = np.average(range_cell, weights=histORGN[_Z[j]:right_cell])  # average calculation for each border
            # mat is a matrix that is initialized in T / F.
            # any value that satisfies the two conditions will get T, otherwise -F
            mat = np.logical_and(imOrig >= _Z[j], imOrig < right_cell)
            _newImg[mat] = _Q[j]  # Where there is a T we will update the new value

        imOr = imOrig / 255.0
        imNew = _newImg / 255.0
        MSE = np.sqrt(np.sum(np.square(np.subtract(imNew, imOr)))) / imOr.size  # According to MSE's function(formula)
        list_MSE.append(MSE)

        if imageIsRGB:  # flag check
            _newImg = _newImg / 255.0
            imageYIQ[:, :, 0] = _newImg
            _newImg = transformYIQ2RGB(imageYIQ)  # Convert image to RGB
        list_quantized.append(_newImg)  # add to quantized_lst

        _Z, _Q = caclculate_middle_boundary(_Z, _Q)  # each boundary become to be a middle of 2 means
        if len(list_MSE) >= 2:
            if np.abs(list_MSE[-1] - list_MSE[-2]) <= 0.000001:
                break

    return list_quantized, list_MSE


