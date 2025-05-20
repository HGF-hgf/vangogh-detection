#!/usr/bin/python

# generate_score_model.py
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
Generate score model
============================================================

Generate score transformation model with PyTorch

"""


import sys
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
from torch_utils import get_model

class ScoreModel(nn.Module):
    def __init__(self, input_size):
        super(ScoreModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--dir', type=dir_type, required=True,
                        help='data directory')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='verbosity level')
    parser.add_argument('-c', '--cores', default=max(1, cpu_count()-2), type=int,
                        choices=range(1, cpu_count()+1),
                        help='number of cores to be used')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='path to import the classifier model')
    parser.add_argument('-s', '--score', type=str, required=True,
                        help='path to export the score model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    args = parser.parse_args(args=argv)
    return args

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

def get_features(model_path, data, device):
    with open(model_path, 'rb') as f:
        saved_model = pickle.load(f)
    
    model = get_model(feature_extract=True)
    model.load_state_dict(saved_model['state_dict'])
    model = model.to(device)
    model.eval()
    
    features = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=32):
            batch = batch.to(device)
            feat = model(batch)
            features.append(feat.cpu())
    
    return torch.cat(features)

def train_score_model(features, classes, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to PyTorch tensors
    features = torch.FloatTensor(features)
    classes = torch.FloatTensor(classes)
    
    # Setup cross-validation
    n_splits = 5
    best_val_loss = float('inf')
    best_model = None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, classes)):
        print_verbose(f"Fold {fold+1}/{n_splits}", 0)
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(features[train_idx], classes[train_idx].unsqueeze(1)),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(features[val_idx], classes[val_idx].unsqueeze(1)),
            batch_size=args.batch_size
        )
        
        # Create model
        model = ScoreModel(features.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    val_loss += criterion(output, y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print_verbose(f'Epoch {epoch+1}/{args.epochs}:', 1)
            print_verbose(f'Train Loss: {train_loss:.4f}', 1)
            print_verbose(f'Val Loss: {val_loss:.4f}', 1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
    
    return {
        'state_dict': best_model,
        'best_loss': best_val_loss,
        'input_size': features.shape[1]
    }

def main(argv):
    args = parse_args(argv)
    set_verbose_level(args.verbose)
    set_n_cores(args.cores)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data and extract features
    data, labels, classes = gen_data(args.dir)
    features = get_features(args.model, torch.FloatTensor(data), device)
    
    # Train score model
    score_model = train_score_model(features, classes, args)
    print_verbose('Best validation loss: %.4f' % score_model['best_loss'], 0)
    
    # Save model
    print_verbose('Saving score model to %s' % args.score, 0)
    with open(args.score, "wb") as f:
        pickle.dump(score_model, f)

if __name__ == "__main__":
    main(sys.argv[1:])
