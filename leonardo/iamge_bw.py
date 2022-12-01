from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt

def toCv2(image):
    """
    converts a Pillow image to OpenCV format
    :param image: the image to convert
    :return: the converted image
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2RGB)

def main(): 

    # open the image to compare in grayscale:
    image = Image.open('images/cubismo_picasso.jpg')
    imageCv2 = toCv2(image)

    # open a new image in grayscale which will contain the polygons:
    width, height = image.size
    numPixels = width * height
    polyImage = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image, 'RGB')
    print(imageCv2)
#run main
if __name__ == "__main__":
    main()
