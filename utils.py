import os
import re
import numpy as np
import imageio.v2 as imageio
from matplotlib import cm
import sys
from glob import glob

FLOW_LIBRARY_PATH = "./data/flow_library"
sys.path.append(FLOW_LIBRARY_PATH)
import flow_IO


BASEPATH_KITTI = "./data/kitti/"
BASEPATH_FT3D = './data/ft3d/'
BASEPATH_SPRING = './data/spring'

__all__ = ['load_kitti_images', 'load_kitti_sf', 'load_ft3d_images',
           'load_ft3d_sf',
           'split_spring_seq', 'BASEPATH_SPRING',
           'load_spring_images', 'load_spring_sf', 'prepare_spring_data_dict']
'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''


def load_pfm(path):
    file = open(path, "r", encoding="latin-1")
    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = data * scale
    return np.reshape(data, shape)


def write_sfl_file(filepath, sf):
    height, width, nBands = sf.shape
    with open(filepath, 'wb') as f:
        f.write('PIEH'.encode())
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        sf.astype(np.float32).tofile(f)
    return


def read_sfl_file(filepath):
    with open(filepath, 'rb') as f:
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == 202021.25, (
            'Flow number %r incorrect. Invalid .sfl file' % flo_number)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=4 * w * h)
        sf = np.resize(data, (h, w, 4))
        return sf


def load_kitti_images(n, dataset='training'):
    path_left = os.path.join(BASEPATH_KITTI, dataset, "image_2")
    path_right = os.path.join(BASEPATH_KITTI, dataset, "image_3")

    impath_left_1 = os.path.join(path_left, "%06d_10.png" % n)
    impath_left_2 = os.path.join(path_left, "%06d_11.png" % n)
    impath_right_1 = os.path.join(path_right, "%06d_10.png" % n)
    impath_right_2 = os.path.join(path_right, "%06d_11.png" % n)

    im_left_1 = imageio.imread(impath_left_1) / 255.0
    im_left_2 = imageio.imread(impath_left_2) / 255.0
    im_right_1 = imageio.imread(impath_right_1) / 255.0
    im_right_2 = imageio.imread(impath_right_2) / 255.0

    return im_left_1, im_right_1, im_left_2, im_right_2


def load_kitti_sf(n, occ='occ'):
    path = os.path.join(BASEPATH_KITTI, "training")

    ofpath = os.path.join(path, "flow_%s/%06d_10.png" % (occ, n))
    d0path = os.path.join(path, "disp_%s_0/%06d_10.png" % (occ, n))
    d1path = os.path.join(path, "disp_%s_1/%06d_10.png" % (occ, n))

    of = imageio.imread(ofpath, format="PNG-FI").astype(float)
    u = (of[:, :, 0] - 2 ** 15) / 64.0
    v = (of[:, :, 1] - 2 ** 15) / 64.0
    d0 = imageio.imread(d0path) / 256.0
    d1 = imageio.imread(d1path) / 256.0

    sf = np.stack([u, v, d0, d1], axis=2)
    sf[d0 <= 0] = 0

    return sf


def load_spring_images(*image_paths):

    img1 = imageio.imread(image_paths[0])/255.0
    img2 = imageio.imread(image_paths[1])/255.0
    img3 = imageio.imread(image_paths[2])/255.0
    img4 = imageio.imread(image_paths[3])/255.0

    return img1, img2, img3, img4


def load_spring_sf(*paths):
    disp1 = flow_IO.readDispFile(paths[0])
    disp2 = flow_IO.readDispFile(paths[1])
    opt_flow = flow_IO.readFlowFile(paths[2])

    return disp1, disp2, opt_flow


def load_ft3d_images(letter, sequence, frame, forward=True, subset='TRAIN'):
    if forward:
        tempstep = 1
    else:
        tempstep = -1

    impath_left_1 = os.path.join(BASEPATH_FT3D, "frames_finalpass",
                                 subset, "%s/%04d/left/%04d.png" % (letter, sequence, frame))
    impath_left_2 = os.path.join(BASEPATH_FT3D, "frames_finalpass", subset,
                                 "%s/%04d/left/%04d.png" % (letter, sequence, frame+tempstep))
    impath_right_1 = os.path.join(BASEPATH_FT3D, "frames_finalpass",
                                  subset, "%s/%04d/right/%04d.png" % (letter, sequence, frame))
    impath_right_2 = os.path.join(BASEPATH_FT3D, "frames_finalpass", subset,
                                  "%s/%04d/right/%04d.png" % (letter, sequence, frame+tempstep))

    im_left_1 = imageio.imread(impath_left_1)[:, :, :3] / 255.0
    im_left_2 = imageio.imread(impath_left_2)[:, :, :3] / 255.0
    im_right_1 = imageio.imread(impath_right_1)[:, :, :3] / 255.0
    im_right_2 = imageio.imread(impath_right_2)[:, :, :3] / 255.0

    return im_left_1, im_right_1, im_left_2, im_right_2


def load_ft3d_sf(letter, sequence, frame, forward=True, subset='TRAIN'):
    if forward:
        of_path = os.path.join(BASEPATH_FT3D, "optical_flow", subset,
                               "%s/%04d/into_future/left/OpticalFlowIntoFuture_%04d_L.pfm" % (letter, sequence, frame))
        dispchange_path = os.path.join(BASEPATH_FT3D, "disparity_change", subset,
                                       "%s/%04d/into_future/left/%04d.pfm" % (letter, sequence, frame))
    else:
        of_path = os.path.join(BASEPATH_FT3D, "optical_flow", subset,
                               "%s/%04d/into_past/left/OpticalFlowIntoPast_%04d_L.pfm" % (letter, sequence, frame))
        dispchange_path = os.path.join(BASEPATH_FT3D, "disparity_change", subset,
                                       "%s/%04d/into_past/left/%04d.pfm" % (letter, sequence, frame))
    disp_path = os.path.join(BASEPATH_FT3D, "disparity", subset,
                             "%s/%04d/left/%04d.pfm" % (letter, sequence, frame))

    flow = np.flipud(load_pfm(of_path))
    disp0 = np.flipud(load_pfm(disp_path))
    disp_change = np.flipud(load_pfm(dispchange_path))
    disp1 = disp0 + disp_change

    sf = np.stack((flow[:, :, 0], flow[:, :, 1], disp0, disp1), axis=2)

    return sf


def colored_flow(flow, maxflow=-1, mask=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxrad = maxflow
    if maxrad < 0:
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    if mask is not None:
        img[mask == 0] = 0
    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - \
        np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC,
               2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - \
        np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM,
               0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def colored_disparity(disp, maxdisp=-1, mask=None):
    maxd = maxdisp
    if maxd < 0:
        maxd = np.max(disp)
    vals = disp/maxd
    img = cm.jet(vals)
    img[vals > 1] = [1, 0, 0, 1]
    if mask is not None:
        img[mask != 1] = [0, 0, 0, 1]
    img = img[:, :, 0:3]
    return np.uint8(img*255)


def prepare_spring_data_dict(root, split):
    '''prepares a dict type mapping between scenes and their respective data.'''
    seq_root = os.path.join(root, 'spring', split)
    scene_dict = {}
    for scene in sorted(os.listdir(seq_root)):
        for cam in ["left", "right"]:
            images = sorted(
                glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
            # self._image_paths.extend(images)
            # forward
            if scene not in scene_dict:
                scene_dict[scene] = []
            for frame in range(1, len(images)):
                scene_dict[scene].append((frame, scene, cam, "FW"))
            # backward
            for frame in reversed(range(2, len(images)+1)):
                scene_dict[scene].append((frame, scene, cam, "BW"))
    return scene_dict


def split_spring_seq(root, split='train', validation_split=0.2):
    '''splits the sequences into training and validation sequences based on the given validation split'''

    def get_seq_to_count(sequences, train_path):
        return {seq: len(os.listdir(os.path.join(train_path, seq, 'frame_left'))) for seq in sequences}

    def get_sum(seq_dict):
        return sum(seq_dict.values())

    # getting sequences
    train_path = os.path.join(root, 'spring', split)
    all_sequences = os.listdir(train_path)
    if split == 'test':
        return all_sequences

    train_seq = all_sequences.copy()
    seq_count = get_seq_to_count(train_seq, train_path)

    # count and get count required validation images
    total_images = get_sum(seq_count)
    num_validation_images = int(validation_split*total_images)

    # sort the sequences in ascending order of the number of images in each sequencees
    sorted_dict = dict(sorted(seq_count.items(), key=lambda value: value[1]))
    val_seq = []
    val_count = 0

    # distributing sequences based on the split ratio
    for seq, count in sorted_dict.items():
        if val_count <= num_validation_images:
            val_seq.append(seq)
            train_seq.remove(seq)
            val_count += count
        # assuring that training split >= validation split
    if validation_split <= 0.5:
        while get_sum(get_seq_to_count(train_seq, train_path)) < get_sum(get_seq_to_count(val_seq, train_path)):
            train_seq.append(val_seq.pop(-1))
            print('Adjusted for traininig dataset to be larger than validation dataset')

        # final checks
    assert len(train_seq) + len(val_seq) == len(
        all_sequences), f"length of train sequences is {len(train_seq)} and {len(val_seq)} while total sequences are {len(all_sequences)}"
    common_seq = list(set(train_seq).intersection(val_seq))
    assert len(
        common_seq) == 0, f"common sequences found in train and validation splits: {common_seq}"
    return train_seq, val_seq


def make_spring_folder(data):
    seq = data[1]
    seq = os.path.join("predictions", seq)
    for cam in ["left", "right"]:
        for direction in ["FW", "BW"]:
            os.makedirs(os.path.join(seq, f'flow_{direction}_{cam}'), exist_ok=True)
            os.makedirs(os.path.join(seq, f'disp2_{direction}_{cam}'), exist_ok=True)
        os.makedirs(os.path.join(seq, f'disp1_{cam}'), exist_ok=True)
        
def write_spring_predictions(prediction, data):
    index, seq, cam, direc = data
    seq = os.path.join("predictions", seq)
    prediction = prediction.squeeze(0)
    flow = prediction[:, :, :2]
    disp1 = prediction[:, :, -2]
    disp2 = prediction[:, :, -1]
    disp1_path = os.path.join(
        seq, f'disp1_{cam}', f'disp1_{cam}_{index:04d}.dsp5')
    disp2_path = os.path.join(
        seq, f'disp2_{direc}_{cam}', f'disp2_{direc}_{cam}_{index:04d}.dsp5')
    flow_path = os.path.join(seq, f'flow_{direc}_{cam}',
                             f'flow_{direc}_{cam}_{index:04d}.flo5')
    flow_IO.writeFlo5File(flow, flow_path)
    flow_IO.writeDsp5File(disp1, disp1_path)
    flow_IO.writeDsp5File(disp2, disp2_path)
