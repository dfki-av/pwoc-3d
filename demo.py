#!/usr/bin/env python3
import numpy as np
import imageio
from tensorflow.python.keras import backend

import utils
from network import Network

# construct model
with backend.get_graph().as_default():
    net = Network(occlusion=True)

# load weights
net.load_weights('./data/pwoc3d-kitti')

# read images
il1 = np.expand_dims(imageio.imread('./data/il1.png') / 255.0, axis=0)
ir1 = np.expand_dims(imageio.imread('./data/ir1.png') / 255.0, axis=0)
il2 = np.expand_dims(imageio.imread('./data/il2.png') / 255.0, axis=0)
ir2 = np.expand_dims(imageio.imread('./data/ir2.png') / 255.0, axis=0)

# predict scene flow
res = net(inputs=(il1, ir1, il2, ir2))

# write result
utils.write_sfl_file('./data/result.sfl', res[0].numpy())
