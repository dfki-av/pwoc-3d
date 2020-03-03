import numpy as np


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
        assert flo_number == 202021.25, ('Flow number %r incorrect. Invalid .sfl file' % flo_number)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=4 * w * h)
        sf = np.resize(data, (h, w, 4))
    return sf
