from torch import nn
from .record import RecordEncoder, RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible

"""
Entrainement en initialisant les états sur les k premières frames puis prédire les suivantes

Online learning LSTM biblio 

"""
class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        @param config: configuration file of the model
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        @param n_class: number of classes (default: 3)
        @param shallow: load a shallow version of RECORD (fewer channels in the decoder)
        """
        super(RecordOI, self).__init__()
        self.encoder = RecordEncoder(config=config['encoder_config'], in_channels=in_channels, norm=norm)
        self.decoder = RecordDecoder(config=config['decoder_config'], n_class=n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD-OI model
        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        st_features_lstm1, st_features_lstm2, st_features_backbone = self.encoder(x)
        confmap_pred = self.decoder(st_features_lstm1, st_features_lstm2, st_features_backbone)
        return self.sigmoid(confmap_pred)

