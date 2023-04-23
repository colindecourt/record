import os
import numpy as np
from datasets import ROD2021Dataset
from datasets.cruw.collate_functions import cr_collate
from cruw.eval.rod.load_txt import read_gt_txt, read_rodnet_res
from cruw.eval.rod.rod_eval_utils import compute_ols_dts_gts, evaluate_img, accumulate, summarize 
from torch.utils.data import DataLoader
import tabulate
from cruw import CRUW

OLSTHRS = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
RECTHRS = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
GT_DIR = '/projets/deeprad/CRUW/ROD2021/annotations/train/'
DICT_FRAME = {'2019_05_09_PBMS004': 900, '2019_09_29_ONRD002': 1692, '2019_05_09_PCMS002': 900, '2019_05_29_PCMS005': 900, '2019_04_30_PCMS001': 900, '2019_04_09_PMS1000': 898, '2019_04_09_PMS3000': 898, '2019_04_09_PMS2000': 898, '2019_05_29_PM3S000': 900, '2019_04_30_PBMS003': 900, '2019_05_23_PM1S015': 900, '2019_05_23_PM1S012': 900, '2019_05_23_PM1S014': 900, '2019_04_09_BMS1000': 897, '2019_05_23_PM2S011': 900, '2019_04_09_BMS1001': 897, '2019_05_29_BCMS000': 900, '2019_04_09_CMS1002': 897, '2019_04_30_MLMS000': 900, '2019_04_09_PMS1001': 898, '2019_04_30_PM2S004': 900, '2019_04_30_PM2S003': 900, '2019_04_09_BMS1002': 897, '2019_05_09_BM1S008': 900, '2019_05_09_CM1S004': 900, '2019_09_29_ONRD011': 1682, '2019_04_30_MLMS001': 900, '2019_05_29_BM1S016': 900, '2019_05_29_PBMS007': 900, '2019_05_23_PM1S013': 900, '2019_09_29_ONRD006': 1695, '2019_09_29_ONRD001': 1698, '2019_05_29_PM2S015': 900, '2019_04_30_MLMS002': 900, '2019_09_29_ONRD005': 1697, '2019_05_09_MLMS003': 900, '2019_05_29_MLMS006': 900, '2019_05_29_BM1S017': 900, '2019_04_30_PBMS002': 900, '2019_09_29_ONRD013': 1690}


def eval_on_val(trainer, executor, dataset, data_root, config_dict, all_confmaps, ckpt_path='best'):
    """
    Evaluate model performances on ROD2021 dataset validation set
    @param trainer: Pytorch Lightning trainer
    @param executor: Pytorch Lightning model executor
    @param dataset: CRUW dataset object
    @param data_root: path to prepared data
    @param config_dict: configuration dictionary of the model
    @param all_confmaps: which confmaps to use (all or only the last timestep)
    @param ckpt_path: path of the checkpoint to load for inference
    """
    val_seqs = sorted(os.listdir(os.path.join(config_dict['dataset_cfg']['data_dir'], 'valid')))
    val_seqs = sorted([seq.split('.')[0] for seq in val_seqs])
    seq_names = sorted(os.listdir(os.path.join(data_root, config_dict['dataset_cfg']['valid']['subdir'])))
    seq_names = sorted([seq for seq in seq_names if seq in val_seqs])
    
    config_dict['train_cfg']['train_stride'] = 1
    for subset in seq_names:
        # Eval current sequence
        gt_path = os.path.join(config_dict['dataset_cfg']['anno_root'], config_dict['dataset_cfg']['valid']['subdir'], subset.upper() + '.txt')
        print(gt_path)
        res_path = os.path.join(trainer.logger.log_dir, 'val', subset.upper() + '.txt')

        data_path = os.path.join(dataset.data_root, 'sequences', config_dict['dataset_cfg']['valid']['subdir'],
                                 gt_path.split('/')[-1][:-4])
        n_frame = len(os.listdir(os.path.join(data_path, dataset.sensor_cfg.camera_cfg['image_folder'])))

        print(n_frame)
        print('Test sequence: {}'.format(subset))
        test_data = ROD2021Dataset(data_dir=config_dict['dataset_cfg']['data_dir'], dataset=dataset,
                                   config_dict=config_dict, split='valid', subset=subset, all_confmaps=all_confmaps)
        
        dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=cr_collate)
        trainer.test(executor, dataloaders=dataloader_test, ckpt_path=ckpt_path)
        
        executor.evaluate_rodnet_seq_(res_path, gt_path, n_frame, subset)

    executor.evaluate_rodnet_()


def eval_on_test(trainer, executor, dataset, data_root, config_dict, all_confmaps, ckpt_path='best'):
    """
    Evaluate model performances on ROD2021 dataset test set
    @param trainer: Pytorch Lightning trainer
    @param executor: Pytorch Lightning model executor
    @param dataset: CRUW dataset object
    @param data_root: path to prepared data
    @param config_dict: configuration dictionary of the model
    @param all_confmaps: which confmaps to use (all or only the last timestep)
    @param ckpt_path: path of the checkpoint to load for inference
    """
    seq_names = sorted(os.listdir(os.path.join(data_root, 'test')))
    for subset in seq_names:
        print('Test sequence: {}'.format(subset))
        data_test = ROD2021Dataset(data_dir=config_dict['dataset_cfg']['data_dir'], dataset=dataset,
                                   config_dict=config_dict, split='test', subset=subset, all_confmaps=all_confmaps)
        
        dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=cr_collate)
        trainer.test(executor, dataloaders=dataloader_test, ckpt_path=ckpt_path)


# Additional functions for evaluation
def summarize_per_class(eval, olsThrs, recThrs, dataset, gl=True, harm=True, per_class=True):
    n_class = dataset.object_cfg.n_class
    dict_per_class = {}

    def _summarize(eval=eval, ap=1, olsThr=None):
        object_counts = eval['object_counts']
        n_objects = np.sum(object_counts)
        metric_per_class = dict()
        if ap == 1:
            # dimension of precision: [TxRxK]
            s = eval['precision']
            # IoU
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:, :, :]
        else:
            # dimension of recall: [TxK]
            s = eval['recall']
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:, :]
        
        if not harm:
            mean_s = np.mean(s[s>-1])
        else:
            mean_s = 0
            for classid in range(n_class):
                if ap == 1:
                    s_class = s[:, :, classid]
                    if len(s_class[s_class > -1]) == 0:
                        pass
                    else:
                        if per_class:
                            metric_per_class[dataset.object_cfg.classes[classid]]= np.mean(s_class[s_class > -1])
                        mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class > -1])
                else:
                    s_class = s[:, classid]
                    if len(s_class[s_class > -1]) == 0:
                        pass
                    else:
                        if per_class:
                            metric_per_class[dataset.object_cfg.classes[classid]]= np.mean(s_class[s_class > -1])
                        mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class > -1])
        
        return mean_s, metric_per_class

    def _summarizeKps():
        stats = np.zeros((12,))
        stats[0], metric_per_class = _summarize(ap=1)
        stats[1], metric_per_class = _summarize(ap=1, olsThr=.5)
        stats[2], metric_per_class = _summarize(ap=1, olsThr=.6)
        stats[3], metric_per_class = _summarize(ap=1, olsThr=.7)
        stats[4], metric_per_class = _summarize(ap=1, olsThr=.8)
        stats[5], metric_per_class = _summarize(ap=1, olsThr=.9)
        stats[6], metric_per_class = _summarize(ap=0)
        stats[7], metric_per_class = _summarize(ap=0, olsThr=.5)
        stats[8], metric_per_class = _summarize(ap=0, olsThr=.6)
        stats[9], metric_per_class = _summarize(ap=0, olsThr=.7)
        stats[10], metric_per_class = _summarize(ap=0, olsThr=.8)
        stats[11], metric_per_class = _summarize(ap=0, olsThr=.9)
        return stats, metric_per_class

    def _summarizeKps_cur():
        stats = np.zeros((2,))
        stats[0], dict_per_class['AP'] = _summarize(ap=1)
        stats[1], dict_per_class['AR'] = _summarize(ap=0)
        return stats, dict_per_class

    if gl:
        summarize = _summarizeKps
    else:
        summarize = _summarizeKps_cur

    stats, dict_per_class = summarize()
    return stats, dict_per_class

