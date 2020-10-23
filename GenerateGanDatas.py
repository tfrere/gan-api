import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import pickle
from PIL import Image, ImageDraw, ImageChops, ImageStat
import base64
import imageio
import json
from io import BytesIO
import math  

class GenerateGanDatas():
    def __init__(self, name):
        tflib.init_tf()
        self.pre_trained_gans = [
            {
                "name": "brains",
                "url": '../input/gan_pre_trained/brains/network-snapshot-010368.pkl'
            },
            {
                "name": "african-masks",
                "url": '../input/gan_pre_trained/masks/network-snapshot-010450.pkl'
            },
            {
                "name": "new-african-masks",
                "url": '../input/gan_pre_trained/new_masks/network-snapshot-010450.pkl'
            },
            {
                "name": "old-photos",
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
        self.name = ""
        self.load_network(name)
        self.name = name  

    def get_pretrained_gan_url_from_name(self, gan_name):
        for i, model in enumerate(self.pre_trained_gans):
            if(gan_name == model["name"]):
                return model["url"]

    def load_network(self, gan_name):
        if(self.name != gan_name):
            model_url = '../input/gan_pre_trained/terre/network-snapshot-010163.pkl'
            model_url = self.get_pretrained_gan_url_from_name(gan_name)
            stream = open(model_url, 'rb')
            with stream:
                self.G, self.D, self.Gs = pickle.load(stream, encoding='latin1')
            self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
            self.name = gan_name

    def generate_zs_from_seeds(self, seeds):
        zs = []
        for seed_idx, seed in enumerate(seeds):
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *self.Gs.input_shape[1:]) # [minibatch, component]
            zs.append(z)
        return zs
            
    @staticmethod
    def from_pil_to_base64_json(pil_image_list):
        base64img_prefix = "data:image/png;base64,"
        final_json = {"images":[]}

        for i, image in enumerate(pil_image_list):
          image.thumbnail((512,512), Image.ANTIALIAS)
          buffered = BytesIO()
          image.save(buffered, format="PNG")
          img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
          final_json["images"].append(base64img_prefix + img_str)

        return final_json

    def get_images_from_zs(self, truncation_psi, zs):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if not isinstance(truncation_psi, list):
            truncation_psi = [truncation_psi] * len(zs)

        imgs = []
        for z_idx, z in enumerate(zs):
            print('Generating image from zs (%d/%d) ...' % (z_idx, len(range(len(zs)))))
            Gs_kwargs.truncation_psi = truncation_psi[z_idx]
            noise_rnd = np.random.RandomState(1) # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
            images = self.Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            imgs.append(Image.fromarray(images[0], 'RGB'))
        return imgs
    
    def get_images_from_seeds(self, truncation_psi, seeds):

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        images_list=[]
        zs=[]
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(range(len(seeds)))))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *self.Gs.input_shape[1:]) # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
            images = self.Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            images_list.append(Image.fromarray(images[0], 'RGB'))
            zs.append(z)

        return images_list, zs

    @staticmethod
    def interpolate(zs, steps):
       out = []
       for i in range(len(zs)-1):
        for index in range(steps):
         fraction = index/float(steps) 
         out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
       return out

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def scalarMultiplyVector(vec, scalar):
      return np.array(vec) * scalar

    @staticmethod
    def calculateDistance(p1,p2) :
      return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def storeCoordinate(xVal, yVal, array):
      array.append({"x":xVal, "y": yVal})

    @staticmethod
    def addVector(v1, v2):
      return v1 + v2

    def combination(self, latents, distances):
      combination = [0] * 512
      for index, distance in enumerate(distances):
        multiplied = self.scalarMultiplyVector(latents[index], distance);
        combination = self.addVector(multiplied, combination)
      return combination

    def getImagesPointsFromDataset(self, numberOfImages, locations, baseImagesDatas):
        
      pointLocationForImages = []

      heightOfCanvas = 1.3
      widthOfCanvas = 1.3

      arrayOfLatent = []
      columns = int(math.sqrt(numberOfImages))

      spbc = widthOfCanvas / columns
      spbl = heightOfCanvas / columns

      for i in range(columns):
        for j in range(columns):
          self.storeCoordinate(i * spbc, j * spbl, pointLocationForImages)

      for i, point in enumerate(pointLocationForImages):
        pointP = [point["x"], point['y']]
        rawDistances = []
        for j, location in enumerate(locations):
          cal = self.calculateDistance(pointP, location)
          rawDistances.append(cal)
        ratios = self.softmax(self.scalarMultiplyVector(rawDistances, -6))
        arrayOfLatent.append(self.combination(baseImagesDatas, ratios))

      return arrayOfLatent


if __name__ == "__main__":
    truncation_psi = 0.7
    seeds = [7069, 6936, 6691, 6679, 6126, 6450]
    test = GenerateGanDatas("african-masks")

    images_list, zs = test.get_images(truncation_psi, seeds)

    print(images_list)