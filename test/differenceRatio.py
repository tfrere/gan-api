# from __future__ import print_function
import json 
import math
import base64
from PIL import Image, ImageChops, ImageStat
from io import BytesIO
import re

# OK 
def diff(img1, img2):
    """
    Calculate the difference between two images of the same size
    by comparing channel values at the pixel level.
    """

    imageData1 = re.sub('^data:image/.+;base64,', '', img1)
    imageData2 = re.sub('^data:image/.+;base64,', '', img2)

    encodedImg1 = BytesIO(base64.b64decode(imageData1))
    encodedImg2 = BytesIO(base64.b64decode(imageData2))

    im1 = Image.open(encodedImg1)
    im2 = Image.open(encodedImg2)

    # Generate diff image in memory.
    diff_img = ImageChops.difference(im1, im2)

    # Calculate difference as a ratio.
    stat = ImageStat.Stat(diff_img)
    # stat.mean can be [r,g,b] or [r,g,b,a].
    removed_channels = 0
    num_channels = len(stat.mean) - removed_channels
    sum_channel_values = sum(stat.mean[:num_channels])
    max_all_channels = num_channels * 255
    diff_ratio = sum_channel_values / max_all_channels

    return diff_ratio


# OK
def lowestIntegerSquaredRoot(value):
    nearestSquaredRoot = math.sqrt(value)
    i = 0
    while nearestSquaredRoot > value:
        nearestSquaredRoot = math.sqrt(value - i)
        i += 1
    return nearestSquaredRoot

# OK
def getIndexInLine(index):
    return index % lineSize

# OK
def getLineOffset(index):
    return int(index // lineSize)

# OK
def isIndexInSquare(index):
    return True if(index > 0 and index < sizeOfDataSet) else False

# OK
def compareBase64(img1, img2):

    encodedImg1 = img1[22:].encode('utf-8')
    encodedImg2 = img2[22:].encode('utf-8')
    b1 = list(base64.b64encode(encodedImg1))
    b2 = list(base64.b64encode(encodedImg2))
    return sum( abs(b1[i] - b2[i]) for i in range(len(b1)))


# reshape in 2d

def compareWithNeighbors(images, index, verbose=False):

    results = []

    if(verbose): print("Comparing -> ", index)

    # determiner les voisins de la case courante
    botNeighborIndex = index + 1 
    topNeighborIndex = index - 1 
    leftNeighborIndex = int(((getLineOffset(index) - 1) * lineSize) + getIndexInLine(index))
    rightNeighborIndex = int(((getLineOffset(index) + 1) * lineSize) + getIndexInLine(index))
    topLeftDiagNeighborIndex = int(((getLineOffset(index) - 1) * lineSize) + getIndexInLine(index) - 1)
    topRightDiagNeighborIndex = int(((getLineOffset(index) + 1) * lineSize) + getIndexInLine(index) - 1)
    botLeftDiagNeighborIndex = int(((getLineOffset(index) - 1) * lineSize) + getIndexInLine(index) + 1)
    botRightDiagNeighborIndex = int(((getLineOffset(index) + 1) * lineSize) + getIndexInLine(index) + 1)

    # calculer le ratio de difference de chaque 
    if isIndexInSquare(botNeighborIndex):
        if(verbose): print("botNeighborIndex-> ", botNeighborIndex)
        results.append(diff(images[index], images[botNeighborIndex]))

    if isIndexInSquare(topNeighborIndex):
        if(verbose): print("topNeighborIndex-> ", topNeighborIndex)
        results.append(diff(images[index], images[topNeighborIndex]))

    if isIndexInSquare(leftNeighborIndex):
        if(verbose): print("leftNeighborIndex-> ", leftNeighborIndex)
        results.append(diff(images[index], images[leftNeighborIndex]))

    if isIndexInSquare(rightNeighborIndex):
        if(verbose): print("rightNeighborIndex-> ", rightNeighborIndex)
        results.append(diff(images[index], images[rightNeighborIndex]))

    if isIndexInSquare(topLeftDiagNeighborIndex):
        if(verbose): print("topLeftDiagNeighborIndex-> ", topLeftDiagNeighborIndex)
        results.append(diff(images[index], images[topLeftDiagNeighborIndex]))

    if isIndexInSquare(topRightDiagNeighborIndex):
        if(verbose): print("topRightDiagNeighborIndex-> ", topRightDiagNeighborIndex)
        results.append(diff(images[index], images[topRightDiagNeighborIndex]))

    if isIndexInSquare(botLeftDiagNeighborIndex):
        if(verbose): print("botLeftDiagNeighborIndex-> ", botLeftDiagNeighborIndex)
        results.append(diff(images[index], images[botLeftDiagNeighborIndex]))

    if isIndexInSquare(botRightDiagNeighborIndex):
        if(verbose): print("botRightDiagNeighborIndex-> ", botRightDiagNeighborIndex)
        results.append(diff(images[index], images[botRightDiagNeighborIndex]))

    # faire la moyenne de ses ratios 
    return sum(results) / len(results)

