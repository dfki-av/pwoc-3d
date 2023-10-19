#!/usr/bin/env python3
import sys
from tensorflow.python.keras import backend
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import datasets
import network
import metrics
import argparse
from utils import write_spring_predictions, make_spring_folder


def eval(checkpoint, data, outstream):
    # construct model
    with backend.get_graph().as_default():
        net = network.Network()

    # load weights
    net.load_weights(checkpoint)

    # prepare metrics
    sf_metric = metrics.SceneFlowMetrics()

    # make predictions
    for e,(images, gt) in enumerate(data):
        print("Evaluating sequence %d..." % e)
        # predict scene flow
        res = net(inputs=images)

        # update metrics
        sf_metric.update_state(gt, res)

    sf_metric.print(stream=outstream)



if __name__ == "__main__":

    arg = sys.argv[1]
    if 'kitti' in arg:
        data = datasets.get_kitti_dataset(datasets.KITTI_VALIDATION_IDXS, batch_size=1)
        eval(arg, data, outstream=sys.stdout)
    elif 'spring' in arg:
        root = datasets.BASEPATH_SPRING
        split = 'train'
        scene_dict = datasets.prepare_spring_data_dict(root, split)
        _, indices = datasets.split_spring_seq(root, split)
        test_dataset = datasets.SpringDataset(root, indices, scene_dict, split)
        data = datasets.get_spring_dataset(test_dataset, 1, split=split)
        eval(arg, data, outstream=sys.stdout)
        
    else:
        raise ValueError(f"checkpoint name should have 'spring' or 'kitti' but given {arg}")
    
