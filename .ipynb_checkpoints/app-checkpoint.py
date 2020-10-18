import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import custom_ops
import re
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw, ImageChops, ImageStat
import base64
import imageio
import json
import pickle

import lib.pretrained_networks
from lib.gan_utility_functions import generate_images_from_seeds
from flask import Flask, jsonify, request

pre_trained_gans = [
    {
        "name": "brains",
        "url": '../input/gan_pre_trained/brains/network-snapshot-010368.pkl'
    },
    {
        "name": "mask",
        "url": '../input/gan_pre_trained/masks/network-snapshot-010450.pkl'
    },
    {
        "name": "old_photos",
        "url": '../input/gan_pre_trained/old_photos/network-snapshot-010491.pkl'
    },
    {
        "name": "chinese",
        "url": '../input/gan_pre_trained/portraits_chinois/network-snapshot-010397.pkl'
    },
    {
        "name": "sneakers",
        "url": '../input/gan_pre_trained/sneakers/network-snapshot-010696.pkl'
    },
    {
        "name": "earth",
        "url": '../input/gan_pre_trained/terre/network-snapshot-010163.pkl'
    },
]

def loadPretrainedGan(gan_name):
    for model in pre_trained_gans:
        if(gan_name == model["gan_name"]):
            return model["url"]

def from_pil_to_base64_json(pil_image_list):
    base64img_prefix = "data:image/png;base64,"
    final_json = {"images":[]}

    for i, image in enumerate(pil_image_list):
      image.thumbnail((256,256), Image.ANTIALIAS)
      buffered = BytesIO()
      image.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
      final_json["images"].append(base64img_prefix + img_str)

    return final_json

app = Flask(__name__)
# app.config["DEBUG"] = True

@app.route('/listPretrainedGans')
def listPretrainedGans():
    return jsonify(pre_trained_gans)

@app.route('/randomImages')
def randomImages():
    print(request.args)
    if 'gan_name' in request.args:
        gan_name = str(request.args['gan_name'])
    else:
        return "Error: No gan_name provided. Please specify it."
    if 'number_of_images' in request.args:
        number_of_images = int(request.args['number_of_images'])
    else:
        return "Error: No number_of_images provided. Please specify it."    

    
    
    seeds = np.random.randint(10000000, size=number_of_images)
    image_list_from_seed = generate_images_from_seeds(seeds, 0.7)
    json_data = from_pil_to_base64_json(image_list_from_seed)
    
    return jsonify(json_data)

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)