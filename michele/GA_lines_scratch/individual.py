import random
import math
import copy
from PIL import Image

class Individual:
    def __init__(self, xSize, ySize, numLines, maxSegLen):
        self.xSize = xSize
        self.ySize = ySize
        self.segLen = maxSegLen
        self.numLines = numLines
        self.matrix = self.__createMatrix(xSize, ySize)
        self.error = None
        self.lines = []

    @staticmethod
    def __createMatrix(xsize, ysize):
        matrix = list()
        for i in range(ysize):
            matrix.append(list())
            for j in range(xsize):
                matrix[i].append(0)
        return matrix

    def __setPixel(self, img, x, y, value):
        img.putpixel((x, y), self.__weightMap(value))

    def matrixToImage(self):
        image = Image.new('L', (self.xSize, self.ySize))
        for y in range(self.ySize):
            for x in range(self.xSize):
                self.__setPixel(image, x, y, self.matrix[y][x])
        # image.show()
        return image

    def calulateError(self, image):
        self.error = 0
        for y in range(self.ySize):
            for x in range(self.xSize):
                self.error += self.__getError(image, x, y)

    def __getError(self, image, x, y):
        return abs(self.__weightMap(self.matrix[y][x]) - image.getpixel((x, y)))

    def __weightMap(self, weight):
        return int(round(255 / (math.exp(float(weight) / 2)), 0))

    def initRandomLines(self):
        self.lines = []
        for i in range(self.numLines):
            self.lines.append(self.__getRandSeg(self.xSize, self.ySize))
        self.__drawLines()

    def crossOver(self, otherParent, mutantPerc, inheritanceRate, refImage):
        assert mutantPerc >= 0
        assert mutantPerc < 1.0
        assert inheritanceRate >= 0
        assert inheritanceRate < 1.0
        child = self.__clone()
        if inheritanceRate > 0:
            n = math.trunc(len(otherParent.lines) * inheritanceRate)
            genetic_heritage = random.sample(otherParent.lines, n)
            toSubstitute = random.sample(range(len(otherParent.lines)), n)
            for gene, idx in zip(genetic_heritage, toSubstitute):
                child.__substSeg(idx, gene, refImage, substIfBetter=False)

        if mutantPerc > 0:
            numLines = len(child.lines)
            numMutants = math.trunc(numLines * mutantPerc)
            if (numMutants == 0):
                numMutants = 1
            for idx in range(numMutants):
                newLine = self.__getRandSeg(self.xSize, self.ySize)
                index = random.randrange(numLines)
                child.__substSeg(index,newLine,refImage,substIfBetter=True)

        return child
        pass

    def __substSeg(self, idx, aLine, refImage, substIfBetter = False):
        deltaErrorOnRemove = self.__removeSeg(self.lines[idx], refImage)
        deltaErrorOnAdd = self.__addSeg(aLine, refImage)
        change = deltaErrorOnRemove+deltaErrorOnAdd
        if substIfBetter and change > 0:
            self.__removeSeg(aLine,refImage)
            self.__addSeg(self.lines[idx],refImage)
        else:
            self.error += change
            self.lines[idx] = aLine

    def __addSeg(self, seg, refImage):
        change = 0
        if abs(seg[0][0] - seg[1][0]) >= abs(seg[0][1] - seg[1][1]):
            start = min(seg[0][0], seg[1][0])
            end = max(seg[0][0], seg[1][0])
            end = min(end, start + self.segLen)
            for x in range(start, end + 1):  # y = ((y1-y0)/(x1-x0))(x - x0) + y0
                y = int(
                    round((float((seg[1][1] - seg[0][1])) / (seg[1][0] - seg[0][0])) * (x - seg[0][0]) + seg[0][1], 0))
                if (y >= 0 and y < self.ySize):
                    oldErr = self.__getError(refImage, x, y)
                    self.matrix[y][x] += 1
                    change += self.__getError(refImage, x, y) - oldErr
        else:
            start = min(seg[0][1], seg[1][1])
            end = max(seg[0][1], seg[1][1])
            end = min(end, start + self.segLen)
            for y in range(start, end + 1):  # x = ((x1-x0)/(y1-y0))(y - y0) + x0
                x = int(
                    round((float((seg[1][0] - seg[0][0])) / (seg[1][1] - seg[0][1])) * (y - seg[0][1]) + seg[0][0], 0))
                if x >= 0 and x < self.xSize:
                    oldErr = self.__getError(refImage, x, y)
                    self.matrix[y][x] += 1
                    change += self.__getError(refImage, x, y) - oldErr
        return change

    def __removeSeg(self, seg, refImage):
        change = 0
        if (abs(seg[0][0] - seg[1][0]) >= abs(seg[0][1] - seg[1][1])):
            start = min(seg[0][0], seg[1][0])
            end = max(seg[0][0], seg[1][0])
            end = min(end, start + self.segLen)
            for x in range(start, end + 1):  # y = ((y1-y0)/(x1-x0))(x - x0) + y0
                y = int(
                    round((float((seg[1][1] - seg[0][1])) / (seg[1][0] - seg[0][0])) * (x - seg[0][0]) + seg[0][1], 0))
                if (y >= 0 and y < self.ySize):
                    oldErr = self.__getError(refImage, x, y)
                    self.matrix[y][x] = max(self.matrix[y][x] - 1, 0)
                    change += self.__getError(refImage, x, y) - oldErr
        else:
            start = min(seg[0][1], seg[1][1])
            end = max(seg[0][1], seg[1][1])
            end = min(end, start + self.segLen)
            for y in range(start, end + 1):  # x = ((x1-x0)/(y1-y0))(y - y0) + x0
                x = int(
                    round((float((seg[1][0] - seg[0][0])) / (seg[1][1] - seg[0][1])) * (y - seg[0][1]) + seg[0][0], 0))
                if (x >= 0 and x < self.xSize):
                    oldErr = self.__getError(refImage, x, y)
                    self.matrix[y][x] = max(self.matrix[y][x] - 1, 0)
                    change += self.__getError(refImage, x, y) - oldErr
        return change

    def __getRandSeg(self, xSize, ySize):
        point1 = (random.randrange(0, xSize), random.randrange(0, ySize))
        point2 = (random.randrange(0, xSize), random.randrange(0, ySize))
        while point2[0] == point1[0] and point2[1] == point1[1]:
            point2 = (random.randrange(0, xSize), random.randrange(0, ySize))
        return point1, point2

    def __drawLines(self):
        for seg in self.lines:
            if abs(seg[0][0] - seg[1][0]) >= abs(seg[0][1] - seg[1][1]):
                start = min(seg[0][0], seg[1][0])
                end = max(seg[0][0], seg[1][0])
                end = min(end, start + self.segLen)
                for x in range(start, end + 1):  # y = ((y1-y0)/(x1-x0))(x - x0) + y0
                    y = int(
                        round((float((seg[1][1] - seg[0][1])) / (seg[1][0] - seg[0][0])) * (x - seg[0][0]) + seg[0][1],
                              0))
                    if (y >= 0 and y < self.ySize):
                        self.matrix[y][x] += 1
            else:
                start = min(seg[0][1], seg[1][1])
                end = max(seg[0][1], seg[1][1])
                end = min(end, start + self.segLen)
                for y in range(start, end + 1):  # x = ((x1-x0)/(y1-y0))(y - y0) + x0
                    x = int(
                        round((float((seg[1][0] - seg[0][0])) / (seg[1][1] - seg[0][1])) * (y - seg[0][1]) + seg[0][0],
                              0))
                    if (x >= 0 and x < self.xSize):
                        self.matrix[y][x] += 1

    def __clone(self):
        return copy.deepcopy(self)
