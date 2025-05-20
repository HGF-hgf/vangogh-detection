#!/usr/bin/python

# classify.py
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
Classify
============================================================

Classify using different aggregation methods with PyTorch

"""


import sys
import argparse
from multiprocessing import cpu_count
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torchvision.models as models

from common import dir_type, print_verbose, set_verbose_level, get_n_cores, set_n_cores
from gather_data import gen_data, parse_class

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

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--dir', type=dir_type, required=True,
                        help='data directory')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='verbosity level')
    parser.add_argument('-c', '--cores', default=get_n_cores(), type=int,
                        choices=range(1, cpu_count()+1),
                        help='number of cores to be used')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='path to import the classifier model')
    parser.add_argument('-s', '--score', type=str,
                        help='path to import the score model')
    parser.add_argument('-a', '--aggregation',
                        choices=['mode','mean','max','far'],
                        default='mean',
                        help='aggregation method (default: mean)')
    parser.add_argument('-g', '--gtruth', action='store_true',
                        help='ground truth class is available (default: False)')


    args = parser.parse_args(args=argv)
    return args


def aggregate_predictions(preds, method='mean'):
    if method == 'mode':
        return torch.mode(preds)[0]
    elif method == 'mean':
        return torch.mean(preds)
    elif method == 'max':
        return torch.max(preds)[0]
    elif method == 'far':
        # Tính khoảng cách giữa các dự đoán
        sorted_preds = torch.sort(preds)[0]
        # Lấy giá trị trung bình của các dự đoán xa nhất
        return torch.mean(sorted_preds[-len(sorted_preds)//4:])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def classify(data, labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load classifier model
    with open(args.model, 'rb') as f:
        saved_model = pickle.load(f)
    
    model = get_model(feature_extract=False)
    model.load_state_dict(saved_model['state_dict'])
    model = model.to(device)
    model.eval()
    
    classification = {}
    unique_labels = np.unique(labels)
    
    print_verbose(f"Number of unique labels: {len(unique_labels)}", 1)
    print_verbose(f"Data shape: {data.shape}", 1)
    
    for label in unique_labels:
        print_verbose(f"Classifying label: {label}", 1)
        mask = labels == label
        label_data = torch.FloatTensor(data[mask]).to(device)
        
        print_verbose(f"Label data shape: {label_data.shape}", 2)
        
        with torch.no_grad():
            outputs = model(label_data)
            predictions = torch.softmax(outputs, dim=1)[:, 1]
            
            print_verbose(f"Predictions shape: {predictions.shape}", 2)
            print_verbose(f"Predictions: {predictions.cpu().numpy()}", 3)
        
        agg_pred = aggregate_predictions(predictions, args.aggregation)
        print_verbose(f"Aggregated prediction: {agg_pred}", 2)
        
        classification[label] = 1 if agg_pred >= 0.5 else 0
    
    print_verbose(f"Classification results: {classification}", 1)
    return classification


def eval_perf(classification):
    from sklearn import metrics
    
    if not classification:
        print_verbose("Warning: No classifications were made!", 0)
        return
    
    y_true = []
    y_pred = []
    
    for key, value in classification.items():
        y_true.append(parse_class(key))
        y_pred.append(value)
    
    if not y_true or not y_pred:
        print_verbose("Warning: No ground truth or predictions available!", 0)
        return
    
    print_verbose("Confusion Matrix:", 0)
    print_verbose(metrics.confusion_matrix(y_true, y_pred), 0)
    print_verbose("Classification Report:", 0)
    print_verbose(metrics.classification_report(y_true, y_pred), 0)


def main(argv):
    args = parse_args(argv)
    set_verbose_level(args.verbose)
    set_n_cores(args.cores)

    print_verbose("Args: %s" % str(args), 1)

    # Get data
    data, labels = gen_data(args.dir, False)
    print_verbose('Data shape: %s' % str(data.shape), 2)
    print_verbose('Labels shape: %s' % str(labels.shape), 2)
    
    # Validate data
    if data.size == 0:
        print_verbose("Error: No data loaded!", 0)
        return
        
    if len(labels) == 0:
        print_verbose("Error: No labels loaded!", 0)
        return
        
    print_verbose(f"Unique labels: {np.unique(labels)}", 1)
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    print_verbose(f"Number of samples per label: {dict(zip(unique_labels, counts))}", 1)

    # Classify
    classification = classify(data, labels, args)
    print_verbose('Final classification: %s' % str(classification), 0)

    # Evaluate performance
    if args.gtruth:
        eval_perf(classification)


if __name__ == "__main__":
    main(sys.argv[1:])
