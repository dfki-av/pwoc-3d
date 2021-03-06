import tensorflow as tf
import numpy as np
from utils import load_kitti_images, load_kitti_sf, load_ft3d_images, load_ft3d_sf
import random

KITTI_MEAN_PIXEL = [0.3791041, 0.39846687, 0.38367166] # RGB
FT3D_MEAN_PIXEL = [0.424101, 0.40341005, 0.36796424] # RGB

KITTI_TRAIN_IDXS = [2, 43, 44, 158, 78, 102, 56, 13, 107, 99, 31, 55, 54, 129, 85, 151, 173, 186, 195, 130, 48, 196, 154, 28, 165,
                    63, 60, 161, 140, 194, 104, 114, 35, 16, 152, 77, 126, 23, 125, 10, 86, 124, 160, 80, 98, 193, 69, 118, 115,
                    30, 92, 134, 71, 57, 8, 178, 38, 182, 27, 67, 36, 139, 91, 6, 49, 179, 184, 84, 81, 188, 101, 5, 141, 166,
                    113, 12, 199, 65, 128, 18, 41, 82, 53, 146, 187, 14, 19, 34, 21, 46, 180, 172, 106, 137, 145, 153, 191, 20,
                    22, 144, 70, 183, 190, 29, 156, 119, 25, 135, 1, 176, 103, 42, 33, 3, 17, 64, 108, 75, 164, 11, 143, 88, 117,
                    26, 4, 162, 177, 83, 73, 171, 109, 111, 15, 50, 100, 181, 167, 148, 79, 168, 76, 94, 121, 89, 198, 68, 138,
                    112, 170, 72, 120, 155, 66, 149, 47, 59, 90, 185, 189, 105, 52, 132, 45, 110, 127, 7, 157, 96, 24, 122, 147,
                    116, 0, 9, 58, 97, 62, 192, 142, 123]
KITTI_VALIDATION_IDXS = [95, 159, 175, 37, 74, 93, 174, 40, 133, 131, 150, 163, 39, 136, 169, 61, 197, 87, 32, 51]
KITTI_DEBUG_IDXS = KITTI_TRAIN_IDXS[:2]
KITTI_TRAIN_SAMPLES = 180
KITTI_VALIDATION_SAMPLES = 20
KITTI_DEBUG_SAMPLES = 2

FT3D_LETTER_LISTS = {
    'A': list(filter(lambda x: x not in [12, 18, 96, 132, 186, 441, 456, 483, 653, 676, 728]+[60, 91, 169, 179, 364, 398, 518, 521, 658], range(750))), # 730
    'B': list(filter(lambda x: x not in [18, 172, 316, 400, 459]+[53, 189, 424, 668], range(750))), # 741
    'C': list(filter(lambda x: x not in [31, 80, 140, 260, 323, 398, 419, 651], range(750))), # 742
}
FT3D_TRAIN_LIST = list(t for t in ((letter, seq, frame) for letter in ['A','B','C'] for seq in FT3D_LETTER_LISTS[letter][:-50] for frame in range(6,16)))
FT3D_VALIDATION_LIST = list(t for t in ((letter, seq, frame) for letter in ['A','B','C'] for seq in FT3D_LETTER_LISTS[letter][-50:] for frame in range(6,16)))
FT3D_TRAINING_SAMPLES = len(FT3D_TRAIN_LIST) # 20630
FT3D_VALIDATION_SAMPLES = len(FT3D_VALIDATION_LIST) # 1500


def _kitti_data_with_labels(idxs):
    for n in idxs:
        images = load_kitti_images(n)
        sf = load_kitti_sf(n)
        yield images, sf


def _ft3d_data_with_labels(dataset_name, shuffle=False, temporal_augmentation=False):
    if dataset_name == 'TRAIN':
        data_list = FT3D_TRAIN_LIST
    elif dataset_name == 'VALID':
        data_list = FT3D_VALIDATION_LIST
    else:
        raise ValueError("Dataset not understood. Select out of 'TRAIN' or 'VALID'.")

    if shuffle:
        np.random.shuffle(data_list)

    for letter, sequence, frame in data_list:

        if (temporal_augmentation and frame > 6 and bool(random.getrandbits(1))) or frame == 15:
            images = load_ft3d_images(letter, sequence, frame, forward=False)
            sf = load_ft3d_sf(letter, sequence, frame, forward=False)
        else:
            images = load_ft3d_images(letter, sequence, frame, forward=True)
            sf = load_ft3d_sf(letter, sequence, frame, forward=True)

        yield images, sf


def _augment(images, sf, vertical_flipping=False):

    stacked_images = tf.stack(images, axis=0)

    # Gaussian noise has a sigma uniformly sampled from [0, 0.04]
    noise = tf.random.normal(shape=tf.shape(stacked_images), mean=0.0, stddev=tf.random.uniform((), 0., 0.04), dtype=tf.float32)
    augmented_images = stacked_images + noise

    # Contrast is sampled within [0.2, 1.4]
    augmented_images = tf.image.adjust_contrast(augmented_images, tf.random.uniform((), 0.2, 1.4))

    # Multiplicative colour changes to the RGB channels per image from [0.5, 2]
    mult = tf.random.uniform((3,), 0.5, 2.)
    augmented_images *= mult
    augmented_images = tf.clip_by_value(augmented_images, 0., 1.)

    # Gamma values from [0.7, 1.5]
    gamma = tf.random.uniform((), 0.7, 1.5)
    augmented_images = tf.image.adjust_gamma(augmented_images, gamma=gamma)

    # Additive brightness changes using Gaussian with a sigma of 0.2
    augmented_images = tf.image.adjust_brightness(augmented_images, tf.random.truncated_normal((), mean=0., stddev=0.2))
    augmented_images = tf.clip_by_value(augmented_images, 0., 1.)

    # Randomly flip the images and ground truth vertically
    if vertical_flipping and bool(random.getrandbits(1)):
        augmented_images = tf.map_fn(tf.image.flip_up_down, augmented_images)
        sf = tf.image.flip_up_down(sf)
        sf *= [1., -1., 1., 1.]

    return (augmented_images[0], augmented_images[1], augmented_images[2], augmented_images[3]), sf


def _random_crop(images, sf, target_size):
    stacked_batch = tf.concat(images+(sf,), axis=2)
    cropped_stack = tf.image.random_crop(stacked_batch, size=target_size+(4*3+4,))
    return (cropped_stack[:,:,0:3], cropped_stack[:,:,3:6], cropped_stack[:,:,6:9], cropped_stack[:,:,9:12]), cropped_stack[:,:,12:]


def get_kitti_dataset(idxs, batch_size, augment=False, shuffle=False, crop=False):
    dataset = tf.data.Dataset.from_generator(lambda: _kitti_data_with_labels(idxs),
                                                output_types=(4*(tf.float32,), tf.float32),
                                                output_shapes=(4*((None, None, 3),), (None, None, 4)))
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(idxs), reshuffle_each_iteration=True)
    if batch_size > 1 or crop:
        dataset = dataset.map(map_func=lambda ims, gt: _random_crop(ims, gt, target_size=(370, 1224)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        dataset = dataset.map(map_func=_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def get_ft3d_dataset(subset, batch_size, augment=False, random_cropping=True, shuffle=False, temporal_augmentation=False):
    dataset = tf.data.Dataset.from_generator(lambda: _ft3d_data_with_labels(subset, shuffle=shuffle, temporal_augmentation=temporal_augmentation),
                                                output_types=(4*(tf.float32,), tf.float32),
                                                output_shapes=(4*((None, None, 3),), (None, None, 4)))
    if random_cropping:
        dataset = dataset.map(map_func=lambda ims, gt: _random_crop(ims, gt, target_size=(512, 960)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        dataset = dataset.map(map_func=lambda ims, gt: _augment(ims, gt, vertical_flipping=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
