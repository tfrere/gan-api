import re
import sys
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import custom_ops
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
import base64
import imageio
from PIL import Image, ImageDraw, ImageChops, ImageStat
import json
from flask import Flask, jsonify, request

from difference_ratio import compareWithNeighborsFrom2dArray, lowestIntegerSquaredRoot
from GenerateGanDatas import GenerateGanDatas

app = Flask(__name__)

has_to_print = False

print("Starting app.py")

def printFile(name, data):
    with open(str(name) + ".json", 'w') as outfile:
        json.dump(data, outfile)
        

GanObject = GenerateGanDatas("african-masks")

# ----------------------------------------------------
@app.route('/listPretrainedGans')
def listPretrainedGans():
    print("ROUTE /listPretrainedGans")
    
    if(has_to_print):
        printFile(request.url[20:], GanObject.pre_trained_gans)
        
    return jsonify(GanObject.pre_trained_gans)

# ----------------------------------------------------
@app.route('/getRandomImages')
def randomImages():
    print("ROUTE /getRandomImages")
    if 'gan_name' in request.args:
        gan_name = str(request.args['gan_name'])
    else:
        return "Error: No gan_name provided. Please specify it."
    if 'number_of_images' in request.args:
        number_of_images = int(request.args['number_of_images'])
    else:
        return "Error: No number_of_images provided. Please specify it."    
    print("for " + str(gan_name))
    print("with number of images -> " + str(number_of_images))
    GanObject.load_network(gan_name)
    seeds = np.random.randint(10000000, size=number_of_images)

    image_list_from_seed, zs = GanObject.get_images_from_seeds(0.7, seeds)
    json_data = GanObject.from_pil_to_base64_json(image_list_from_seed)
    json_data["seeds"] = seeds.tolist()
    
    if(has_to_print):
        printFile(request.url[20:], json_data)
    
    return jsonify(json_data)

# ----------------------------------------------------
@app.route('/getImageInterpolationFromSeeds')
def generateImageInterpolationFromSeed():
    print("ROUTE /getImageInterpolationFromSeeds")
    if 'gan_name' in request.args:
        gan_name = str(request.args['gan_name'])
    else:
        return "Error: No gan_name provided. Please specify it."
    if 'number_of_images' in request.args:
        number_of_images = int(request.args['number_of_images'])
    else:
        return "Error: No gan_name provided. Please specify it."
    if 'seeds' in request.args:
        seeds = request.args.getlist('seeds')
    else:
        return "Error: No seeds provided. Please specify it."
    
    print("for " + gan_name)
    GanObject.load_network(gan_name)
    
    seeds = list(map(int, seeds))
    print("with seeds ->")
    print(seeds)
    print("with number_of_images ->")
    print(number_of_images)
    
    zs = GanObject.generate_zs_from_seeds(seeds)
    zs_interpolation = GanObject.interpolate(zs, number_of_images // (len(seeds) - 1))
    imgs = GanObject.get_images_from_zs(1.0, zs_interpolation)
    json_data = GanObject.from_pil_to_base64_json(imgs)
    
    if(has_to_print):
        printFile(request.url[20:], json_data)
    
    return(jsonify(json_data))

# ----------------------------------------------------
@app.route('/get2dMapFromSeeds')
def get2dMapFromSeeds():
    print("ROUTE /get2dMapFromSeeds")
    if 'gan_name' in request.args:
        gan_name = str(request.args['gan_name'])
    else:
        return "Error: No gan_name provided. Please specify it."
    if 'number_of_images' in request.args:
        number_of_images = int(request.args['number_of_images'])
    else:
        return "Error: No gan_name provided. Please specify it."
    if 'seeds' in request.args:
        seeds = request.args.getlist('seeds')
    else:
        return "Error: No seeds provided. Please specify it."
    
    
    print("for " + gan_name)
    GanObject.load_network(gan_name)
    
    seeds = list(map(int, seeds))
    print("with seeds ->")
    print(seeds)
    print("with number_of_images ->")
    print(number_of_images)
    
    image_list_from_seed, zs = GanObject.get_images_from_seeds(0.7, seeds)
    
    coords_to_test = [[0.71, 1.14],[1.16, 1.02],[1.23, 0.54],[0.71, 0.25],[0.22, 0.53],[0.29, 1.05]]
    result = GanObject.getImagesPointsFromDataset(number_of_images, coords_to_test, zs)
    
    images = []
    for i, image in enumerate(result):
        images.append(np.array(image, dtype=np.float64).reshape(1,512))
        
    imgs = GanObject.get_images_from_zs(1.0, images)
    
    base64img_prefix = "data:image/png;base64,"
    final_json = {
        "baseImgLocations": coords_to_test,
        "differenceRatios": [],
        "images":[]
    }

    for i, image in enumerate(imgs):
      print("Transforming into base64 image -> " + str(i))
      image.thumbnail((256,256), Image.ANTIALIAS)

      buffered = BytesIO()
      image.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
      final_json["images"].append(base64img_prefix + img_str)

    sizeOfDataSet = len(final_json["images"])
    print(sizeOfDataSet)

    lineSize = lowestIntegerSquaredRoot(sizeOfDataSet)
    
    np_array = np.array(final_json["images"])
    two_dimensional_array = np.reshape(np_array, (-1, int(lineSize)))

    for index, image in enumerate(final_json["images"]):
        print("Getting differences for neighbors -> " + str(index) + " / " + str(sizeOfDataSet))
        final_json["differenceRatios"].append(compareWithNeighborsFrom2dArray(two_dimensional_array, index, verbose=False))
    
    # for debugging purpose, we can hide images base64 hash to avoid flooding the shell
    # final_json["images"] = []
    
    if(has_to_print):
        printFile(request.url[20:], final_json)
    
    return jsonify(final_json)

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
# ----------------------------------------------------
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)
