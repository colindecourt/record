"""
Inspired from https://github.com/yizhou-wang/RODNet
"""
import matplotlib.pyplot as plt
import numpy as np
from .object_class import get_class_name
import wandb

def log_preds(ra_map, confmap_gt, confmap_pred, image_path, 
                model_name='RRODNet', set='train'):
    
    if confmap_gt is not None:
        fig, axes = plt.subplots(1, 4, figsize=(9, 3), tight_layout=True)
        confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

    confmap_pred = np.transpose(confmap_pred, (1, 2, 0))
    image = plt.imread(image_path)

    axes[0].imshow(image)
    axes[0].set_title('Camera')
    axes[0].set_axis_off()
    
    axes[1].imshow(ra_map, cmap='plasma')
    axes[1].set_title('Radar (RA)')
    axes[1].set_axis_off()

    axes[2].imshow(confmap_pred)
    axes[2].set_title(model_name+' prediction')
    axes[2].set_axis_off()
    
    if confmap_gt is not None:
        axes[3].imshow(confmap_gt)
        axes[3].set_title('Ground truth')
        axes[3].set_axis_off()

    plt.tight_layout()
    wandb.log({set: fig})
    plt.clf()

def log_preds_with_dets(res_final, ra_map, confmap_gt, confmap_pred, 
                        image_path, cruw_dataset, model_name='RRODNet', set='train'):

    max_dets, _ = res_final.shape
    classes = cruw_dataset.object_cfg.classes

    if confmap_gt is not None:
        fig, axes = plt.subplots(1, 4, figsize=(9, 3), tight_layout=True)
        confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

    confmap_pred = np.transpose(confmap_pred, (1, 2, 0))
    image = plt.imread(image_path)

    axes[0].imshow(image)
    axes[0].set_title('Camera')
    axes[0].set_axis_off()

    axes[1].imshow(ra_map, cmap='plasma')
    axes[1].set_title('Radar (RA)')
    axes[1].set_axis_off()
    
    axes[2].imshow(confmap_pred)
    for d in range(max_dets):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        plt.scatter(col_id, row_id, s=10, c='white')
        text = cla_str + '\n%.2f' % conf
        plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    axes[2].set_title(model_name+' prediction')
    axes[2].set_axis_off()
    
    if confmap_gt is not None:
        axes[3].imshow(confmap_gt)
        axes[3].set_title('Ground truth')
        axes[3].set_axis_off()

    wandb.log({set: fig})
    plt.clf()