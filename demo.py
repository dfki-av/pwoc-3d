#!/usr/bin/env python3
import numpy as np
import imageio
from tensorflow.python.keras import backend

import utils
import network

# construct model
with backend.get_graph().as_default():
    net = network.Network(occlusion=True)

# load weights
net.load_weights('./data/pwoc3d-kitti')

# read images
il1 = np.expand_dims(imageio.imread('./data/il1.png') / 255.0, axis=0)
ir1 = np.expand_dims(imageio.imread('./data/ir1.png') / 255.0, axis=0)
il2 = np.expand_dims(imageio.imread('./data/il2.png') / 255.0, axis=0)
ir2 = np.expand_dims(imageio.imread('./data/ir2.png') / 255.0, axis=0)

# predict scene flow
res = net(inputs=(il1, ir1, il2, ir2))[0].numpy()

# save visualization
imageio.imwrite("./data/viz_optical_flow.png", utils.colored_flow(res[:,:,:2]))
imageio.imwrite("./data/viz_disparity1.png", utils.colored_disparity(res[:,:,2]))
imageio.imwrite("./data/viz_disparity2.png", utils.colored_disparity(res[:,:,3], maxdisp=np.max(res[:,:,2])))

# write result
utils.write_sfl_file('./data/result.sfl', res)
