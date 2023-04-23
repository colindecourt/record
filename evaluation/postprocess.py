"""
From: https://github.com/yizhou-wang/RODNet
"""
from .lnms import lnms
import numpy as np
from utils.object_class import get_class_name


def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col


def post_process_single_frame_cruw(confmaps, dataset, config_dict):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    n_class = dataset.object_cfg.n_class
    rng_grid = dataset.range_grid
    agl_grid = dataset.angle_grid
    model_configs = config_dict['model_cfg']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']

    class_size, height, width = confmaps.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = confmaps[c, :, :]
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = dict(
                frame_id=None,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=c,
                score=conf,
            )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = lnms(obj_dicts_in_class, dataset, config_dict)
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    res_final[:, :] = detect_mat[:max_dets]

    return res_final


def write_dets_results_single_frame(res, data_id, save_path, dataset):
    max_dets, _  = res.shape
    classes = dataset.object_cfg.classes
    with open(save_path, 'a+') as f:
        for d in range(max_dets):
            cla_id = int(res[d, 0])
            if cla_id == -1:
                continue
            row_id = res[d, 1]
            col_id = res[d, 2]
            conf = res[d, 3]
            f.write("%d %s %d %d %s\n" % (data_id, get_class_name(cla_id, classes), row_id, col_id, conf))