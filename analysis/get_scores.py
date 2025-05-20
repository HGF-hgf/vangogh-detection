#!/usr/bin/python

# get_scores.py
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
Get scores
============================================================

Calculate score probabilities

"""


import sys
import argparse
from multiprocessing import cpu_count
import numpy as np
import pickle
import torch
import torch.nn as nn

from common import dir_type, file_type, print_verbose, set_verbose_level, get_verbose_level, get_n_cores, set_n_cores

from gather_data import gen_data


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--dir', type=dir_type, required=True,
                        help='data directory')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='verbosity level')
    parser.add_argument('-c', '--cores', default=get_n_cores(), type=int,
                        choices=range(1, cpu_count()+1),
                        help='number of cores to be used (default: %d)' % get_n_cores())
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='path to import the classifier model')
    parser.add_argument('-s', '--score', type=str, required=True,
                        help='path to import the score model')
    parser.add_argument('-t', '--targets', type=file_type, required=True,
                        help='input file with target labels')
    parser.add_argument('-n', '--number', type=int, default=2,
                        help='number of patches to analyze')


    args = parser.parse_args(args=argv)
    return args


def calc_prob(data, labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read models
    with open(args.model, "rb") as f:
        clf_model = pickle.load(f)
    print_verbose("Classifier Model loaded", 4)

    with open(args.score, "rb") as f:
        score_model = pickle.load(f)
    print_verbose("Score Model loaded", 4)

    # Create and load model
    model = nn.Sequential(
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)  # 2 classes: van Gogh and non-van Gogh
    )
    model.load_state_dict(clf_model['state_dict'])
    model = model.to(device)
    model.eval()

    # Read targets
    with open(args.targets) as f:
        targets = f.read().splitlines()

    # Get results
    for t in targets:
        print_verbose("Calculating label %s ..." % t, 0)
        
        # Find all patches that start with this label
        mask = np.array([label.startswith(t) for label in labels])
        if not np.any(mask):
            print_verbose("Warning: No data found for label %s" % t, 0)
            continue
            
        label_data = torch.FloatTensor(data[mask]).to(device)
        print_verbose("Found %d patches for label %s" % (len(label_data), t), 1)
        
        with torch.no_grad():
            outputs = model(label_data)
            predictions = torch.softmax(outputs, dim=1)[:, 1]
            predictions = predictions.cpu().numpy()
            
        # Sort predictions
        sorted_preds = np.sort(predictions)
        
        # First N predictions
        first_preds = sorted_preds[:args.number]
        print_verbose("First %d scores for %s:\n%s" % (args.number, t, str(first_preds)), 0)
        
        # Last N predictions
        last_preds = sorted_preds[-args.number:]
        print_verbose("Last %d scores for %s:\n%s" % (args.number, t, str(last_preds)), 0)

    return 0


def main(argv):
    # Parse arguments
    args = parse_args(argv)
    set_verbose_level(args.verbose)
    set_n_cores(args.cores)

    print_verbose("Args: %s" % str(args), 1)

    # Some tests
    data, labels = gen_data(args.dir, False)

    print_verbose('Data: %s' % str(data), 5)
    print_verbose('Labels: %s' % str(labels), 4)

    print_verbose('Data shape: %s' % str(data.shape), 2)
    print_verbose('Labels shape: %s' % str(labels.shape), 2)

    calc_prob(data, labels, args)


if __name__ == "__main__":
    main(sys.argv[1:])
