# PWOC-3D

Official TensorFlow 2 implementation of our paper "[PWOC-3D: Deep Occlusion-Aware End-to-End Scene Flow Estimation](https://av.dfki.de/publications/pwoc-3d-deep-occlusion-aware-end-to-end-scene-flow-estimation/)" (IV, 2019)

## Getting Started
Use one of the download scripts (`download.sh` or `download.ps1`) to download and extract the [pre-trained weights](https://cloud.dfki.de/owncloud/index.php/s/DEqe5SQCxSGWRkQ/download) of PWOC-3D, or do it manually.

Execute `demo.py` to estimate scene flow for the sample in the `data` folder. The output will be saved as `result.sfl` along with visualizations for the optical flow and disparity components into the same folder.

### Requirements
- Python 3.x
    - [TensorFlow](https://www.tensorflow.org/) >= 2.0.0
    - [TensorFlow Addons](https://github.com/tensorflow/addons)
    - NumPy
    - ImageIO
    - Matplotib (for visualization only)
    - tqdm (for training only)

You can install all dependencies using pip: `pip install -r requirements.txt`

Successfully tested environments:\
Windows 10: Python 3.6.7, TensorFlow 2.1.0, Numpy 1.17.4, ImageIO 2.6.1\
Ubuntu 18.04: Python 3.6.8, TensorFlow 2.1.0, Numpy 1.17.4, ImageIO 2.8.0
 

### Data Format
The output file of the demo is in `.sfl` format. This is the straight forward extension of middlebury `.flo` file format for optical flow to scene flow. `.sfl` files are binary files according the following format:

```
".sfl" file format used to store scene flow
Stores 4-band float image for flow and disparity components.
Floats are stored in little-endian order.
An optical flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
A disparity value is considered "unknown" if it is less or equal 0.
A scene flow value is considered "unknown" if either ot the flow or disparity components is unknown.
```

|bytes   |contents                                                                                                                                                                                                                              |
|--------|--------------------------------------------------------------------------------------------------------------|
|0-3     |tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25 <br>(just a sanity check that floats are represented correctly)|
|4-7     |width as an integer|
|8-11    |height as an integer|
|12-end  |data (width * height * 4 * 4 bytes total) the float values for u and v, d0, and d1 interleaved, in row order, i.e. <br>u[row0,col0], v[row0,col0], d0[row0,col0], d1[row0,col0], <br> u[row0,col1], v[row0,col1], d0[row0,col1], d1[row0,col1], ... |

## Training a model
We provide code for training our model from scratch or using the pre-trained weights for FlyingThings3D. Make sure you have downloaded the required data into `./data/kitti/` or `./data/ft3d/`, or change the `BASEPATH` variables in `utils.py` to point to the location of the data.

For pre-training from scratch on FlyingThings3D execute `train.py --pretrain`.

For fine-tuning on KITTI based on the initial weights for FlyingThings3D execute `train.py --finetune`. Make sure you have pre-trained the model on FlyingThings3D or downloaded the pre-trained weights.

If you really plan to train from scratch, it is advisable that you train the model without the occlusion estimation module first (i.e. using the flag `-noocc`). The occlusion estimator follows the idea of self-attention and thus might be instable when features are less distinctive right in the beginning of training. A typical training routine from scratch till a fine-tuned KITTI model thus might look like this:
```
python train.py --pretrain --noocc
python train.py --pretrain --init_with="./models/pwoc3d-ft3d-noocc/pwoc3d-ft3d-noocc"
python train.py --finetune --init_with="./models/pwoc3d-ft3d/pwoc3d-ft3d"
```   

###Monitoring the training
During training, results will be logged as tensorboard summaries.

Run `tensorboard --logdir=./summaries` to see the graphs for all common scene flow metrics.

## Citation
If you find the code or the paper useful, consider citing us:
```
@inproceedings{saxena2019pwoc,
  title={{PWOC-3D}: Deep Occlusion-Aware End-to-End Scene Flow Estimation},
  author={Saxena, Rohan and Schuster, Ren{\'e} and Wasenm{\"u}ller, Oliver and Stricker, Didier},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  year={2019},
}
```
