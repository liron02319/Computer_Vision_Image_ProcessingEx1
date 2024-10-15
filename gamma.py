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
import cv2
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    image = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:  # check if the image grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image from BGR TO GRAY
    title_window = 'Display Gamma'
    trackbar_name = 'Gamma'
    cv2.namedWindow(title_window)   # namedWindow
    cv2.createTrackbar(trackbar_name, title_window, 0, 100, on_trackbar)    # createTrackbar
    while True:
        gamma = cv2.getTrackbarPos(trackbar_name, title_window)  # trackbar position.
        gamma = gamma / 100 * (2 - 0.01)  # calculate for  slider’s values with resolution 0.01(from 0-2)
        gamma = 0.01 if gamma == 0 else gamma  # check the gamma value
        newImage = correction_of_gamma(image, gamma)
        cv2.imshow(title_window, newImage)
        t = cv2.waitKey(1000)
        if t == 27:  # if press esc button
            break
        if cv2.getWindowProperty(title_window, cv2.WND_PROP_VISIBLE) < 1:  # check the WindowProperty
            break
    cv2.destroyAllWindows()  # close the windows


def on_trackbar(k: int):  # for function createTrackbar parameter
    pass

def correction_of_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
        Gamma correction
        :param image: the original image
        :param gamma: the gamma number
        :return: the new image after the gamma operation
        """
    powerGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** powerGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")    # calculate
    return cv2.LUT(image, table)



def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
