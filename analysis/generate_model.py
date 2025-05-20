#!/usr/bin/python

# generate_model.py
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
Generate model
============================================================

Generate classification model with PyTorch

"""


import sys
import os
import argparse
from multiprocessing import cpu_count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import pickle

from common import dir_type, print_verbose, set_verbose_level, get_n_cores, set_n_cores
from gather_data import gen_data



def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--dir', type=dir_type, required=True,
                        help='data directory')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='verbosity level')
    parser.add_argument('-c', '--cores', default=int(cpu_count()-2), type=int,
                        choices=range(1, cpu_count()+1),
                        help='number of cores to be used')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='path to export the generated model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    args = parser.parse_args(args=argv)
    return args


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def get_model(feature_extract=False):
    """Create a simple neural network for feature classification"""
    model = nn.Sequential(
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)  # 2 classes: van Gogh and non-van Gogh
    )
    return model


def generate_model(data, classes, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert data to PyTorch tensors
    data = torch.FloatTensor(data)
    classes = torch.LongTensor(classes)
    
    # Setup cross-validation
    n_splits = 5
    best_val_acc = 0
    best_model = None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, classes)):
        print_verbose(f"Fold {fold+1}/{n_splits}", 0)
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(data[train_idx], classes[train_idx]),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(data[val_idx], classes[val_idx]),
            batch_size=args.batch_size
        )
        
        # Create model
        model = get_model(feature_extract=False)
        model = model.to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Training loop
        for epoch in range(args.epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            
            print_verbose(f'Epoch {epoch+1}/{args.epochs}:', 1)
            print_verbose(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%', 1)
            print_verbose(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%', 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict()
    
    return {
        'state_dict': best_model,
        'best_acc': best_val_acc,
        'input_size': data.shape[1]
    }


def main(argv):

    # Parse arguments
    args = parse_args(argv)
    set_verbose_level(args.verbose)
    set_n_cores(args.cores)

    print_verbose("Args: %s" % str(args), 1)

    # Training
    data, labels, classes = gen_data(args.dir)
    print_verbose('Data shape: %s' % str(data.shape), 2)
    print_verbose('Classes shape: %s' % str(classes.shape), 2)

    model = generate_model(data, classes, args)
    print_verbose('Best validation accuracy: %.2f%%' % model['best_acc'], 0)

    # Export
    print_verbose('Saving model to %s' % args.model, 0)
    with open(args.model, "wb") as f:
        pickle.dump(model, f)

    print_verbose('Done!', 0)


if __name__ == "__main__":
    main(sys.argv[1:])
