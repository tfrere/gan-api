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
def getIndexInLine(index, lineSize):
    return index % lineSize

# OK
def getLineOffset(index, lineSize):
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

getNeighbors = lambda x, y, lineSize : [(x2, y2) for x2 in range(x-1, x+2)
                            for y2 in range(y-1, y+2)
                            if (-1 < x <= lineSize-1 and
                                -1 < y <= lineSize-1 and
                                (x != x2 or y != y2) and
                                (0 <= x2 <= lineSize-1) and
                                (0 <= y2 <= lineSize-1))]

# neighbors = lambda x, y : [(x+a[0], y+a[1]) for a in 
#                     [(-1,0), (1,0), (0,-1), (0,1)] 
#                     if ( (0 <= x+a[0] < w) and (0 <= y+a[1] < h))]

def compareWithNeighborsFrom2dArray(images, index, verbose=False):

    results = []
    lineSize = len(images[0])
    line = getLineOffset(index, lineSize)
    indexInLine = getIndexInLine(index, lineSize)
    active_neighbors = getNeighbors(line, indexInLine, lineSize)

    for i, neighbor in enumerate(active_neighbors):
        results.append(diff(images[line][indexInLine], images[neighbor[0]][neighbor[1]]))

    return sum(results) / len(results)