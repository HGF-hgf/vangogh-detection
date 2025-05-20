#!/usr/bin/python

"""
============================================================
PyTorch utilities for feature extraction and model handling
============================================================
"""

import torch
import torchvision.transforms as transforms
from torchvision import models

def get_model(model_name='vgg16', weights=models.VGG16_Weights.IMAGENET1K_V1, feature_extract=True):
    """
    Load a PyTorch model.
    
    Args:
        model_name (str): Name of the model to load
        weights: Model weights to use (default: VGG16_Weights.IMAGENET1K_V1)
        feature_extract (bool): Whether to use the model as feature extractor
    
    Returns:
        model: PyTorch model
    """
    if model_name.lower() == 'vgg16':
        model = models.vgg16(weights=weights)
        if feature_extract:
            model = torch.nn.Sequential(*list(model.features),
                                      model.avgpool,
                                      torch.nn.Flatten(1),
                                      *list(model.classifier)[:-1])
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

def get_transform(is_train=False):
    """
    Get standard PyTorch transforms for image preprocessing.
    
    Args:
        is_train (bool): Whether to include augmentation for training
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    from common import IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMAGE_SIZE
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(DEFAULT_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(DEFAULT_IMAGE_SIZE + 32),
            transforms.CenterCrop(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    return transform 