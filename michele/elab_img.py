from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("images/tower.jpg")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(imgray, 10, 250)

filename = 'images/savedImage.jpg'
cv2.imwrite(filename, edged)

# edged = cv2.Canny(imgray, 10, 250)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
# closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#
# (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0*peri, True)
#     cv2.drawContours(imgray, [approx], -1, (0,255,0), 2)

# image = Image.new('L', (480, 360))
# draw = ImageDraw.Draw(image, 'L')
# img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


cv2.imshow('image', edged)
cv2.waitKey(0)
