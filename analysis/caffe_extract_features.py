#!/usr/bin/python

# caffe_extract_features.py
# Copyright 2016
#   Guilherme Folego (gfolego@gmail.com)
#   Otavio Gomes (otaviolmiro@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
============================================================
Extract Features (PyTorch)
============================================================

Extract features from images using a PyTorch model.
Default using VGG model with ImageNet normalization:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

References:
  https://pytorch.org/
  https://pytorch.org/hub/pytorch_vision_vgg/
"""


import sys
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from common import dir_type, file_type, \
    print_verbose, set_verbose_level, get_verbose_level

BATCH_SIZE = 32

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', type=str, default='vgg16',
                        help='model name (default: vgg16)')
    parser.add_argument('-l', '--list', type=file_type, required=True,
                        help='file containing list of images to process')
    parser.add_argument('-i', '--input', type=dir_type, required=True,
                        help='input images directory')
    parser.add_argument('-o', '--output', type=dir_type, required=True,
                        help='output features directory')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='verbosity level')

    args = parser.parse_args(args=argv)
    return args

def main(argv):
    # Parse arguments
    args = parse_args(argv)
    set_verbose_level(args.verbose)
    print_verbose("Args: %s" % str(args), 1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if args.model.lower() == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Remove the last classifier layer to get features
        model = torch.nn.Sequential(*list(model.features),
                                  model.avgpool,
                                  torch.nn.Flatten(1),
                                  *list(model.classifier)[:-1])
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    model = model.to(device)
    model.eval()

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Read image names
    with open(args.list) as f:
        allnames = f.read().splitlines()

    # Process images in batches
    for sub in range(0, len(allnames), BATCH_SIZE):
        fnames = allnames[sub : sub+BATCH_SIZE]
        batch = []

        # Preprocess images
        for fname in fnames:
            fpath = os.path.join(args.input, fname)
            print(f"Processing image {fpath} ...")
            
            image = Image.open(fpath).convert('RGB')
            input_tensor = preprocess(image)
            batch.append(input_tensor)

        # Create batch tensor
        batch = torch.stack(batch).to(device)

        # Extract features
        print("Extracting features ...")
        with torch.no_grad():
            features = model(batch)

        # Write extracted features
        for idx, fname in enumerate(fnames):
            path = os.path.join(args.output, os.path.dirname(fname))
            if not os.path.exists(path):
                os.makedirs(path)
            fpath = os.path.join(args.output, fname + ".feat")
            print(f"Writing features to {fpath} ...")
            np.savetxt(fpath, features[idx].cpu().numpy())

    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])

