import torch
from torchinfo import summary
import yaml

def test_record_cruw():
    from models import Record
    config = yaml.load(open('../configs/cruw/config_record_cruw.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load RECORD ----')
    model = Record(config=backbone_config, n_class=3, in_channels=8)
    print('---- OK ----')

    dummy_input = torch.zeros((1, 8, 12, 128, 128))
    print('---- Model summary ----')
    summary(model, input_size=dummy_input.shape)

    print('---- Test model with a dummy input of shape (1, 8, 16, 128, 128)')
    dummy_output = model(dummy_input.cuda())
    assert dummy_output.shape == (1, 3, 128, 128)
    print('---- OK ----')

    config = yaml.load(open('../configs/cruw/config_record_s_cruw.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load RECORD-S ----')
    model = Record(config=backbone_config, n_class=3, in_channels=8)
    print('---- OK ----')
    print('---- Model summary ----')
    summary(model, input_size=dummy_input.shape)

    print('---- Test model with a dummy input of shape (1, 8, 16, 128, 128)')
    dummy_output = model(dummy_input.cuda())
    assert dummy_output.shape == (1, 3, 128, 128)
    print('---- OK ----')

def test_record_carrada():
    dummy_input_rd = torch.rand((1, 1, 5, 256, 64))
    dummy_input_ad = torch.rand((1, 1, 5, 256, 64))
    dummy_input_ra = torch.rand((1, 1, 5, 256, 256))
    from models import MVRecord
    config = yaml.load(open('../configs/carrada/config_mvrecord_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load MV-RECORD ----')
    model = MVRecord(config=backbone_config, n_classes=4, n_frames=5)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=[dummy_input_rd.shape, dummy_input_ra.shape, dummy_input_ad.shape])

    print('---- Test model with a dummy RAD input')
    dummy_output = model(dummy_input_rd.cuda(), dummy_input_ra.cuda(), dummy_input_ad.cuda())
    assert dummy_output[0].shape == (1, 4, 256, 64)
    assert dummy_output[1].shape == (1, 4, 256, 256)

    config = yaml.load(open('../configs/carrada/config_mvrecord_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load MV-RECORD-S ----')
    model = MVRecord(config=backbone_config, n_classes=4, n_frames=5)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=[dummy_input_rd.shape, dummy_input_ra.shape, dummy_input_ad.shape])

    print('---- Test model with a dummy RAD input')
    dummy_output = model(dummy_input_rd.cuda(), dummy_input_ra.cuda(), dummy_input_ad.cuda())
    assert dummy_output[0].shape == (1, 4, 256, 64)
    assert dummy_output[1].shape == (1, 4, 256, 256)

    from models import Record
    config = yaml.load(open('../configs/carrada/config_record_ra_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load RECORD-RA ----')
    model = Record(config=backbone_config, n_class=4, in_channels=1)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=dummy_input_ra.shape)

    print('---- Test model with a dummy RA input')
    dummy_output = model(dummy_input_ra.cuda())
    assert dummy_output.shape == (1, 4, 256, 256)

    config = yaml.load(open('../configs/carrada/config_record_rd_carrada.yaml', 'rb'), yaml.FullLoader)
    backbone_config = yaml.load(open(config['model_cfg']['backbone_pth'], 'rb'), yaml.FullLoader)
    print('---- Load RECORD-RD ----')
    model = Record(config=backbone_config, n_class=4, in_channels=1)
    print('---- OK ----')

    print('---- Model summary ----')
    summary(model, input_size=dummy_input_rd.shape)

    print('---- Test model with a dummy RD input')
    dummy_output = model(dummy_input_rd.cuda())
    assert dummy_output.shape == (1, 4, 256, 64)
