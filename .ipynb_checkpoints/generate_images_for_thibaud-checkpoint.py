seeds = [7069, 6936, 6691, 6679, 6126, 6450]

import argparse
import numpy as np
import PIL
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import pickle
from lib.pretrained_networks import load_networks
from dnnlib.tflib import custom_ops

tflib.init_tf()
model = '../input/gan_pre_trained/brains/network-snapshot-010368.pkl'
truncation_psi = 0.7
nb = 6
_G, _D, Gs = load_networks(model)
def get_images(truncation_psi, nb, seeds):
    stream = open(model, 'rb')
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    images_list=[]
    zs=[]
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(range(nb))))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        images_list.append(images)
        zs.append(z)
        
    return np.array(images_list), np.array(zs)


images_list, zs = get_images(truncation_psi, nb, seeds)


print(images_list)