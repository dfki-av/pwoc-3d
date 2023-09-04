import tensorflow as tf
import numpy as np
from utils import *
import random
from typing import List
import os
from tqdm import tqdm
import json


KITTI_MEAN_PIXEL = [0.3791041, 0.39846687, 0.38367166]  # RGB
FT3D_MEAN_PIXEL = [0.424101, 0.40341005, 0.36796424]  # RGB
SPRING_MEAN_PIXEL = [0.21417567, 0.2714196, 0.29755503]  # RGB

KITTI_TRAIN_IDXS = [2, 43, 44, 158, 78, 102, 56, 13, 107, 99, 31, 55, 54, 129, 85, 151, 173, 186, 195, 130, 48, 196, 154, 28, 165,
                    63, 60, 161, 140, 194, 104, 114, 35, 16, 152, 77, 126, 23, 125, 10, 86, 124, 160, 80, 98, 193, 69, 118, 115,
                    30, 92, 134, 71, 57, 8, 178, 38, 182, 27, 67, 36, 139, 91, 6, 49, 179, 184, 84, 81, 188, 101, 5, 141, 166,
                    113, 12, 199, 65, 128, 18, 41, 82, 53, 146, 187, 14, 19, 34, 21, 46, 180, 172, 106, 137, 145, 153, 191, 20,
                    22, 144, 70, 183, 190, 29, 156, 119, 25, 135, 1, 176, 103, 42, 33, 3, 17, 64, 108, 75, 164, 11, 143, 88, 117,
                    26, 4, 162, 177, 83, 73, 171, 109, 111, 15, 50, 100, 181, 167, 148, 79, 168, 76, 94, 121, 89, 198, 68, 138,
                    112, 170, 72, 120, 155, 66, 149, 47, 59, 90, 185, 189, 105, 52, 132, 45, 110, 127, 7, 157, 96, 24, 122, 147,
                    116, 0, 9, 58, 97, 62, 192, 142, 123]
KITTI_VALIDATION_IDXS = [95, 159, 175, 37, 74, 93, 174, 40,
                         133, 131, 150, 163, 39, 136, 169, 61, 197, 87, 32, 51]
KITTI_DEBUG_IDXS = KITTI_TRAIN_IDXS[:2]
KITTI_TRAIN_SAMPLES = 180
KITTI_VALIDATION_SAMPLES = 20
KITTI_DEBUG_SAMPLES = 2

FT3D_LETTER_LISTS = {
    # 730
    'A': list(filter(lambda x: x not in [12, 18, 96, 132, 186, 441, 456, 483, 653, 676, 728]+[60, 91, 169, 179, 364, 398, 518, 521, 658], range(750))),
    # 741
    'B': list(filter(lambda x: x not in [18, 172, 316, 400, 459]+[53, 189, 424, 668], range(750))),
    # 742
    'C': list(filter(lambda x: x not in [31, 80, 140, 260, 323, 398, 419, 651], range(750))),
}
FT3D_TRAIN_LIST = list(t for t in ((letter, seq, frame) for letter in [
                       'A', 'B', 'C'] for seq in FT3D_LETTER_LISTS[letter][:-50] for frame in range(6, 16)))
FT3D_VALIDATION_LIST = list(t for t in ((letter, seq, frame) for letter in [
                            'A', 'B', 'C'] for seq in FT3D_LETTER_LISTS[letter][-50:] for frame in range(6, 16)))
FT3D_TRAINING_SAMPLES = len(FT3D_TRAIN_LIST)  # 20630
FT3D_VALIDATION_SAMPLES = len(FT3D_VALIDATION_LIST)  # 1500

# TODO add comment explaing the below code
if os.path.exists(os.path.join(BASEPATH_SPRING, 'spring', 'train')):
    SPRING_SCENE_DICT = prepare_spring_data_dict(BASEPATH_SPRING, 'train')
    SPRING_TRAINING_IDXS, SPRING_VALIDATION_IDXS = split_spring_seq(
        BASEPATH_SPRING, validation_split=0.25)
    with open("./spring_splits.json", "w+") as f:
        json.dump({'spring_train_idxs': SPRING_TRAINING_IDXS,
                   'spring_val_idxs': SPRING_VALIDATION_IDXS}, f, indent=4)


def _kitti_data_with_labels(idxs):
    # cam_signal determinges whether its left-right disparity or right-left disparity
    # for uniformity among all the dataloaders
    cam_signal = 1 
    for n in idxs:
        images = load_kitti_images(n)
        sf = load_kitti_sf(n)
        yield (images, cam_signal), sf


def _ft3d_data_with_labels(dataset_name, shuffle=False, temporal_augmentation=False):
    # for uniformity among all the dataloaders
    cam_signal = 1 
    if dataset_name == 'TRAIN':
        data_list = FT3D_TRAIN_LIST
    elif dataset_name == 'VALID':
        data_list = FT3D_VALIDATION_LIST
    else:
        raise ValueError(
            "Dataset not understood. Select out of 'TRAIN' or 'VALID'.")

    if shuffle:
        np.random.shuffle(data_list)

    for letter, sequence, frame in data_list:

        if (temporal_augmentation and frame > 6 and bool(random.getrandbits(1))) or frame == 15:
            images = load_ft3d_images(letter, sequence, frame, forward=False)
            sf = load_ft3d_sf(letter, sequence, frame, forward=False)
        else:
            images = load_ft3d_images(letter, sequence, frame, forward=True)
            sf = load_ft3d_sf(letter, sequence, frame, forward=True)

        yield (images, cam_signal), sf


def _augment(images, sf, vertical_flipping=False):


    images, cam_signal = images
    stacked_images = tf.stack(images, axis=0)

    # Gaussian noise has a sigma uniformly sampled from [0, 0.04]
    noise = tf.random.normal(shape=tf.shape(
        stacked_images), mean=0.0, stddev=tf.random.uniform((), 0., 0.04), dtype=tf.float32)
    augmented_images = stacked_images + noise

    # Contrast is sampled within [0.2, 1.4]
    augmented_images = tf.image.adjust_contrast(
        augmented_images, tf.random.uniform((), 0.2, 1.4))

    # Multiplicative colour changes to the RGB channels per image from [0.5, 2]
    mult = tf.random.uniform((3,), 0.5, 2.)
    augmented_images *= mult
    augmented_images = tf.clip_by_value(augmented_images, 0., 1.)

    # Gamma values from [0.7, 1.5]
    gamma = tf.random.uniform((), 0.7, 1.5)
    augmented_images = tf.image.adjust_gamma(augmented_images, gamma=gamma)

    # Additive brightness changes using Gaussian with a sigma of 0.2
    augmented_images = tf.image.adjust_brightness(
        augmented_images, tf.random.truncated_normal((), mean=0., stddev=0.2))
    augmented_images = tf.clip_by_value(augmented_images, 0., 1.)

    # Randomly flip the images and ground truth vertically
    if vertical_flipping and bool(random.getrandbits(1)):
        augmented_images = tf.map_fn(tf.image.flip_up_down, augmented_images)
        sf = tf.image.flip_up_down(sf)
        sf *= [1., -1., 1., 1.]

    images = (augmented_images[0], augmented_images[1],
              augmented_images[2], augmented_images[3])
    
    images = (images, cam_signal)

    return images, sf


def _random_crop(images, sf, target_size):

    images, cam_signal = images
 
    stacked_batch = tf.concat(images+(sf,), axis=2)
    cropped_stack = tf.image.random_crop(
        stacked_batch, size=target_size+(4*3+4,))

    sf = cropped_stack[:, :, 12:]
    images = (cropped_stack[:, :, 0:3], cropped_stack[:, :, 3:6],
              cropped_stack[:, :, 6:9], cropped_stack[:, :, 9:12])

    images = (images, cam_signal)

    return images, sf


def get_kitti_dataset(idxs, batch_size, augment=False, shuffle=False, crop=False):

    output_types = ((4*(tf.float32,), tf.int32), tf.float32)

    dataset = tf.data.Dataset.from_generator(lambda: _kitti_data_with_labels(idxs),
                                             output_types=output_types)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(idxs), reshuffle_each_iteration=True)
    if batch_size > 1 or crop:
        dataset = dataset.map(map_func=lambda ims, gt: _random_crop(
            ims, gt, target_size=(370, 1224)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        dataset = dataset.map(
            map_func=_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def get_ft3d_dataset(subset, batch_size, augment=False, random_cropping=True, shuffle=False, temporal_augmentation=False):

    output_types = ((4*(tf.float32,), tf.int32), tf.float32)
    dataset = tf.data.Dataset.from_generator(lambda: _ft3d_data_with_labels(subset, shuffle=shuffle, temporal_augmentation=temporal_augmentation),
                                             output_types=output_types
                                             )
    if random_cropping:
        dataset = dataset.map(map_func=lambda ims, gt: _random_crop(
            ims, gt, target_size=(512, 960)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        dataset = dataset.map(map_func=lambda ims, gt: _augment(
            ims, gt, vertical_flipping=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


class SpringDataset:

    """
    A Spring Dataset class that can be used to obatin train and test datasets.
    """

    def __init__(self, root: str, indices, data_dict, split: str = 'train', subsample_groundtruth: bool = True, shuffle: bool = False):
        """
        :param: root: Path where the spring/original exits (including spring/original_
        :param: split: type of split. supports only test and train (not validation. validation is controlled by indices)
        :param: subsample_groundtruth: when set to True, subsamples the ground truth by 4 times.
        :param: shuffle: shuffles the given indices.
        """

        self._split = split.lower()
        self._subsample_gt = subsample_groundtruth

        if self._split.startswith('val'):
            split = 'train'
        self.seq_root = os.path.join(root, 'spring', split)
        self._shuffle = shuffle

        self._scene_dict = data_dict
        self._indices = indices

    def __len__(self):
        c = 0
        for index in self._indices:
            frame_data = self._scene_dict[index]
            for data in frame_data:
                c += 1
        return c

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._indices)

        for index in self._indices:
            frame_data = self._scene_dict[index]
            for data in frame_data:
                frame, scene, cam, direction = data

                # reference frame
                img1_path = os.path.join(
                    self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
                # print(img1_path)

                if cam == "left":
                    othercam = "right"
                    cam_signal = 1
                else:
                    cam_signal = -1
                    othercam = "left"

                if direction == "FW":
                    othertimestep = frame+1
                else:
                    othertimestep = frame-1

                # same time step, other cam
                img2_path = os.path.join(
                    self.seq_root, scene, f"frame_{othercam}", f"frame_{othercam}_{frame:04d}.png")
                # other time step, same cam
                img3_path = os.path.join(
                    self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{othertimestep:04d}.png")
                # other time step, other cam
                img4_path = os.path.join(
                    self.seq_root, scene, f"frame_{othercam}", f"frame_{othercam}_{othertimestep:04d}.png")
                img1, img2, img3, img4 = load_spring_images(img1_path,
                                                            img2_path, img3_path, img4_path)
                img1 = self.filter_inf_nan(img1)
                img2 = self.filter_inf_nan(img2)
                img3 = self.filter_inf_nan(img3)
                img4 = self.filter_inf_nan(img4)

                if self._split == "test":
                    yield img1, img2, img3, img4, data
                else:

                    disp1_path = os.path.join(
                        self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
                    disp2_path = os.path.join(
                        self.seq_root, scene, f"disp2_{direction}_{cam}", f"disp2_{direction}_{cam}_{frame:04d}.dsp5")
                    flow_path = os.path.join(
                        self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")

                    disp1, disp2, flow = load_spring_sf(
                        disp1_path, disp2_path, flow_path)

                    if self._subsample_gt:
                        # use only every second value in both spatial directions ==> ground truth will have same dimensions as images
                        disp1 = disp1[::2, ::2]
                        disp2 = disp2[::2, ::2]
                        flow = flow[::2, ::2]
                    sf = np.stack(
                        (flow[:, :, 0], flow[:, :, 1], disp1, disp2), axis=-1)
                    sf = self.filter_inf_nan(sf)
                    yield ((img1, img2, img3, img4), cam_signal), sf

    @staticmethod
    def filter_inf_nan(data):
        data[np.isnan(data) | np.isinf(data)] = 0
        return data


def get_spring_dataset(spring_dataset: SpringDataset,
                       batch_size: int, split='train',
                       augment: bool = False,
                       crop: bool = False,
                       cache_path=None):

    output_types = ((4*(tf.float32,), tf.int32), tf.float32)

    if split == 'test':
        output_types = (tf.float32,
                        tf.float32, tf.float32, tf.float32,
                        (tf.int32, tf.string, tf.string, tf.string))

    dataset = tf.data.Dataset.from_generator(lambda: spring_dataset,
                                             output_types=output_types)

    # if shuffle:
    #     dataset = dataset.shuffle(len(idxs), reshuffle_each_iteration=True)

    if batch_size > 1 or crop:
        dataset = dataset.map(map_func=lambda ims, gt: _random_crop(
            ims, gt, target_size=(370, 1224)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        dataset = dataset.map(
            map_func=_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.cache(cache_path)

    dataset = dataset.batch(
        batch_size, drop_remainder=False, num_parallel_calls=8)
    dataset = dataset.prefetch(100)
    if cache_path:
        dataset = dataset.cache(cache_path)
    return dataset
