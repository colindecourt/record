import random
import colorsys
from cruw.mapping.ops import ra2xzidx_interpolate
from cruw import CRUW
import yaml
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from cruw.mapping import rf2rfcart, pol2cart, pol2cart_ramap


def generate_colors_code(number_of_colors):
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              for i in range(number_of_colors)]
    return colors


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors_rgb(n_colors):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 0.7
    hsv = [(i / n_colors, 1, brightness) for i in range(n_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [list(color) for color in colors]
    return colors


#paths_lists = ["/home/nxf67149/record_cvpr2023/model_zoo/ROD2021/RECORD/val/2019_05_29_MLMS006.txt",
#               "/home/nxf67149/record_cvpr2023/model_zoo/ROD2021/RECORD/val/2019_04_09_PMS1000.txt",
#               "/home/nxf67149/record_cvpr2023/model_zoo/ROD2021/RECORD/val/2019_04_30_MLMS001.txt",
#paths_lists= ["/home/nxf67149/record_cvpr2023/model_zoo/ROD2021/RECORD/val/2019_09_29_ONRD005.txt"]
paths_lists = []

for path_pred in paths_lists:
    print(path_pred)
    seq_name = path_pred.split('/')[-1].split('.')[0]
    sample = os.path.join("/opt/dataset_ssd/radar1/CRUW/data/valid/", seq_name + '.pkl')
    data = pkl.load(open(sample, "rb"))
    images_paths = data['image_paths']
    radar_paths = data['radar_paths']

    with open(path_pred, 'r') as f:
        preds = f.readlines()

    xz_grid = dataset.xz_grid
    frame_id_list = []
    start_id = 1161
    end_id = 1361
    cur_frame_id = start_id
    for pred in preds:
        pred = pred.split(' ')
        frame_id = int(pred[0])
        if frame_id <= end_id and frame_id >= start_id:
            class_name = pred[1]
            r = int(pred[2])
            a = int(pred[3])
            conf = float(pred[4])
            if cur_frame_id == frame_id:
                frame_id_list.append([class_name, r, a, conf])
            else:
                fig = plt.figure(figsize=(8, 3))
                ax = fig.add_subplot(121)
                image = plt.imread(images_paths[frame_id])
                ax.imshow(image)
                ax.axis('off')
                ax.set_title('Camera image', fontsize=10)
                rad = np.load(radar_paths[frame_id][0])
                rf_cart, (x_line, z_line) = rf2rfcart(rad, dataset.range_grid, dataset.angle_grid, dataset.xz_grid)
                ax = fig.add_subplot(122)
                for i, pred_tmp in enumerate(frame_id_list):
                    class_name, r, a, conf = pred_tmp
                    if conf > 0.3:
                        (x, z) = ra2xzidx_interpolate(dataset.range_grid[r], dataset.angle_grid[a], dataset.xz_grid)
                        colors = generate_colors_rgb(60)
                        ax.scatter(int(x), int(z), s=20, color=colors[i])
                        ax.text(int(x) + 2, int(z) + 2, '%s' % class_name, c='white')

                ax.imshow(rf_cart, origin="lower")
                ax.set_xticks(np.arange(0, len(xz_grid[0]), 30))
                ax.set_xticklabels(xz_grid[0][::30], fontsize=8.5)
                ax.set_yticks(np.arange(0, len(xz_grid[1]), 20))
                ax.set_yticklabels(xz_grid[1][::20], fontsize=8.5)
                ax.set_xlabel('x(m)', fontsize=9)
                ax.set_ylabel('z(m)', fontsize=9)
                ax.set_title('Cartesian RA map', fontsize=10)
                plt.savefig('/home/nxf67149/record_cvpr2023/video/' + seq_name + '_' + str(frame_id - 1) + '.png', transparent=True,
                            facecolor='white', bbox_inches='tight')
                plt.close()
                cur_frame_id += 1
                frame_id_list = []
                class_name = pred[1]
                r = int(pred[2])
                a = int(pred[3])
                conf = float(pred[4])
                frame_id_list.append([class_name, r, a, conf])

import imageio
frames = []
for file in sorted(os.listdir("/home/nxf67149/record_cvpr2023/video")):
    if file != ".ipynb_checkpoints":
        image = imageio.v2.imread(os.path.join("/home/nxf67149/record_cvpr2023/video", file))
        frames.append(image)
        print(file)


imageio.mimsave('./example.gif', # output gif
                frames,          # array of input frames
                fps = 20)