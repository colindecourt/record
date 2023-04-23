from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import yaml 


def get_models(config):
    """
    Return the model to train and the corresponding dataset format
    @param config: configuration dictionary to build model
    @return: model
    """
    name = config['name']
    if name in ('RECORD', 'RECORD-RD', 'RECORD-RA'):
        from models import Record
        backbone_cfg = yaml.load(open(config['backbone_pth']), yaml.FullLoader)
        model = Record(config=backbone_cfg, in_channels=config['in_channels'], norm=config['norm'],
                       n_class=config['nb_classes'])
    elif name in ('RECORD-OI', 'RECORD-RD-OI', 'RECORD-RA-OI'):
        from models import RecordOI
        backbone_cfg = yaml.load(open(config['backbone_pth']), yaml.FullLoader)
        model = RecordOI(config=backbone_cfg, in_channels=config['in_channels'], norm=config['norm'],
                       n_class=config['nb_classes'])
    elif name == 'MV-RECORD':
        from models import MVRecord
        backbone_cfg = yaml.load(open(config['backbone_pth']), yaml.FullLoader)
        model = MVRecord(config=backbone_cfg, n_classes=config['nb_classes'], n_frames=config['win_size'],
                                  in_channels=config['in_channels'], norm=config['norm'])
    elif name == 'MV-RECORD-OI':
        from models import MVRecordOI
        backbone_cfg = yaml.load(open(config['backbone_pth']), yaml.FullLoader)
        model = MVRecordOI(config=backbone_cfg, n_classes=config['nb_classes'], n_frames=config['win_size'],
                         in_channels=config['in_channels'], norm=config['norm'])
    elif name in ('RECORDNoLstmMulti', 'RECORDNoLstmSingle'):
        from models import RecordNoLstm
        backbone_cfg = yaml.load(open(config['backbone_pth']), yaml.FullLoader)
        model = RecordNoLstm(backbone_cfg, config['in_channels'], config['nb_classes'])
    else:
        raise ValueError
    return model


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)