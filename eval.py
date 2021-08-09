#!/usr/bin/env python3
import sys
from tensorflow.python.keras import backend

import datasets
import network
import metrics


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
    data = datasets.get_kitti_dataset(datasets.KITTI_VALIDATION_IDXS, batch_size=1)
    eval(sys.argv[1], data, outstream=sys.stdout)
