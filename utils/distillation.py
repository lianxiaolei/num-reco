# coding:utf8

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Distillation(object):
    """

    """
    _origin: np.ndarray

    def __init__(self, img):
        self._origin = img

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, img):
        self._origin = img

    def fft(self, amplitude_thres=None):
        """

        :param amplitude_thres:
        :return:
        """
        f = np.fft.fft2(self.origin)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift))
        magnitude_uint = magnitude.astype(np.uint8)
        if not amplitude_thres:
            return magnitude_uint

        thresval, img_thresh = cv2.threshold(magnitude_uint,
                                             amplitude_thres,
                                             255,
                                             cv2.THRESH_BINARY)
        return img_thresh

    def hlp(self, img, rho, theta, votes, min_length, max_gap):
        print('houghline', rho, theta, votes, min_length, max_gap)
        lines = cv2.HoughLinesP(img, rho, theta, votes, min_length, max_gap)
        return lines

    def rotation_revise(self, rho, theta, votes, min_length, max_gap, amp):
        """

        :param rho:
        :param theta:
        :param votes:
        :param min_length:
        :param max_gap:
        :return:
        """
        img_thresh = self.fft(amp)
        plt.imshow(img_thresh)
        plt.show()

        lines = self.hlp(img_thresh, rho, theta, votes, min_length, max_gap)
        print('lines', lines)

        h, w = self.origin.shape[:2]

        x1, y1, x2, y2 = lines[0][0]
        theta = (y2 - y1) / (x2 - x1)

        angle = math.atan(theta)  # arctan(theta)
        angle = angle * (180 / np.pi)
        angle = (angle + 90)

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(self.origin, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def remove_lines(self, margin=0):
        """

        :param margin:
        :return:
        """
        lines_h = cv2.HoughLinesP(self.origin, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=0)
        lines_v = cv2.HoughLinesP(self.origin, 1, np.pi, 30, minLineLength=60, maxLineGap=0)

        line_img = np.ones((self.origin.shape[0], self.origin.shape[1], 3), dtype=np.uint8)
        line_img *= 255

        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            # theta = (y2 - y1) / (x2 - x1 + 1e-4)

            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if margin:
                for i in range(margin):
                    cv2.line(line_img, (x1, y1 - i), (x2, y2 - i), (0, 255, 0), 1)
                    cv2.line(line_img, (x1, y1 + i), (x2, y2 + i), (0, 255, 0), 1)

        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            # theta = (y2 - y1) / (x2 - x1 + 1e-4)

            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if margin:
                for i in range(margin):
                    cv2.line(line_img, (x1 - i, y1), (x2 - i, y2), (0, 255, 0), 1)
                    cv2.line(line_img, (x1 + i, y1), (x2 + i, y2), (0, 255, 0), 1)

        line_img = 255 - line_img
        self.origin[line_img > 0] = 0
