from torch import nn
from .record import RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual

class RecordEncoderNoLstm(nn.Module):
    def __init__(self, in_channels, config, norm='layer'):
        """
        RECurrent Online object detectOR (RECORD) features extractor.
        @param in_channels: number of input channels (default: 8)
        @param config: number of input channels per block
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(RecordEncoderNoLstm, self).__init__()
        self.norm = norm
        # Set the number of input channels in the configuration file
        config['in_conv']['in_channels'] = in_channels

        # config_tmp = **config['in_conv'] --> dict with arguments

        # Input convolution (expands the number of input channels)
        self.in_conv = Conv3x3ReLUNorm(in_channels=config['in_conv']['in_channels'],
                                       out_channels=config['in_conv']['out_channels'],
                                       stride=config['in_conv']['stride'], norm=norm)

        # IR block 1 (acts as a bottleneck)
        self.ir_block1 = self._make_ir_block(in_channels=config['ir_block1']['in_channels'],
                                             out_channels=config['ir_block1']['out_channels'],
                                             num_block=config['ir_block1']['num_block'],
                                             expansion_factor=config['ir_block1']['expansion_factor'],
                                             stride=config['ir_block1']['stride'], use_norm=config['ir_block1']['use_norm'])

        # IR block 2 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block2 = self._make_ir_block(in_channels=config['ir_block2']['in_channels'],
                                             out_channels=config['ir_block2']['out_channels'],
                                             num_block=config['ir_block2']['num_block'],
                                             expansion_factor=config['ir_block2']['expansion_factor'],
                                             stride=config['ir_block2']['stride'], use_norm=config['ir_block2']['use_norm'])

        # IR block 3 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block3 = self._make_ir_block(in_channels=config['ir_block3']['in_channels'],
                                             out_channels=config['ir_block3']['out_channels'],
                                             num_block=config['ir_block3']['num_block'],
                                             expansion_factor=config['ir_block3']['expansion_factor'],
                                             stride=config['ir_block3']['stride'], use_norm=config['ir_block3']['use_norm'])


        # IR block 4 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block4 = self._make_ir_block(in_channels=config['ir_block4']['in_channels'],
                                             out_channels=config['ir_block4']['out_channels'],
                                             num_block=config['ir_block4']['num_block'],
                                             expansion_factor=config['ir_block4']['expansion_factor'],
                                             stride=config['ir_block4']['stride'], use_norm=config['ir_block4']['use_norm'])


    def forward(self, x):
        """
        @param x: input tensor for timestep t with shape (B, C, H, W)
        @return: list of features maps and hidden states (spatio-temporal features)
        """
        # Extracts spatial information
        x = self.in_conv(x)
        x = self.ir_block1(x)
        x1 = self.ir_block2(x)
        x2 = self.ir_block3(x1)
        x3 = self.ir_block4(x2)

        return x3, x2, x1

    def _make_ir_block(self, in_channels, out_channels, num_block, expansion_factor, stride, use_norm):
        """
        Build an Inverted Residual bottleneck block
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param num_block: number of IR layer in the block
        @param expansion_factor: expansion factor of each IR layer
        @param stride: stride of the first convolution
        @return a torch.nn.Sequential layer
        """
        if use_norm:
            norm = self.norm
        else:
            norm = None
        layers = [InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                   expansion_factor=expansion_factor, norm=norm)]
        for i in range(1, num_block):
            layers.append(InvertedResidual(in_channels=out_channels, out_channels=out_channels, stride=1,
                                           expansion_factor=expansion_factor,  norm=norm))
        return nn.Sequential(*layers)


class RecordNoLstm(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        @param config: configuration file of the model
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        @param n_class: number of classes (default: 3)
        """
        super(RecordNoLstm, self).__init__()
        self.encoder = RecordEncoderNoLstm(config=config['encoder_config'], in_channels=in_channels, norm=norm)
        self.decoder = RecordDecoder(config=config['decoder_config'], n_class=n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD-OI model
        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        x3, x2, x1 = self.encoder(x)
        confmap_pred = self.decoder(x3, x2, x1)
        return self.sigmoid(confmap_pred)

