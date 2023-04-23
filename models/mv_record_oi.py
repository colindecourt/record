import torch
from torch import nn
from models.record_oi import RecordEncoder, RecordDecoder


class MVRecord(nn.Module):
    def __init__(self, config, n_frames, in_channels=1, n_classes=4, norm='layer'):
        """
        Multi view RECurrent Online object detectOR (MV-RECORD) model class
        @param config: config dict to build the model
        @param n_frames: number of input frames (i.e. timesteps)
        @param in_channels: number of input channels (default: 1)
        @param n_classes: number of classes (default: 4)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(MVRecord, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_frames = n_frames

        # Backbone (encoder)
        self.rd_encoder = RecordEncoder(in_channels=self.in_channels, config=config['encoder_rd_config'], norm=norm)
        self.ra_encoder = RecordEncoder(in_channels=self.in_channels, config=config['encoder_ra_config'], norm=norm)
        self.ad_encoder = RecordEncoder(in_channels=self.in_channels, config=config['encoder_ad_config'], norm=norm)

        # Temporal Multi View Skip Connections
        in_channels_skip_connection_lstm1 = config['encoder_rd_config']['bottleneck_lstm1']['in_channels'] + \
                                            config['encoder_ad_config']['bottleneck_lstm1']['in_channels'] + \
                                            config['encoder_ra_config']['bottleneck_lstm1']['in_channels']
        # We project the concatenation of features to the initial #channels of each view (kernel_size = 1)
        self.skip_connection_lstm1_conv = nn.Conv2d(in_channels=in_channels_skip_connection_lstm1,
                                                    out_channels=config['encoder_rd_config']['bottleneck_lstm1']['out_channels'],
                                                    kernel_size=1)

        in_channels_skip_connection_lstm2 = config['encoder_rd_config']['bottleneck_lstm2']['in_channels'] + \
                                            config['encoder_ad_config']['bottleneck_lstm2']['in_channels'] + \
                                            config['encoder_ra_config']['bottleneck_lstm2']['in_channels']
        # We project the concatenation of features to the initial #channels of each view (kernel_size = 1)
        self.skip_connection_lstm2_conv = nn.Conv2d(in_channels=in_channels_skip_connection_lstm2,
                                                    out_channels=config['encoder_rd_config']['bottleneck_lstm2']['out_channels'],
                                                    kernel_size=1)

        # We downsample the RA view on the azimuth dimension to match the size of AD and RD view
        self.down_sample_ra_view_skip_connection1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.up_sample_rd_ad_views_skip_connection1 = nn.Upsample(scale_factor=(1, 2))

        # Decoding
        self.rd_decoder = RecordDecoder(config=config['decoder_rd_config'], n_class=self.n_classes)
        self.ra_decoder = RecordDecoder(config=config['decoder_ra_config'], n_class=self.n_classes)

    def forward(self, x_rd, x_ra, x_ad):
        """
        Forward pass MV-RECORD model
        @param x_rd: RD input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @param x_ra: RA input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @param x_ad: AD input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: RD and RA segmentation masks of the last time step with shape (B, n_class, H, W)
        """
        # Backbone
        st_features_backbone_rd, st_features_lstm2_rd, st_features_lstm1_rd = self.rd_encoder(x_rd)
        st_features_backbone_ra, st_features_lstm2_ra, st_features_lstm1_ra = self.ra_encoder(x_ra)
        st_features_backbone_ad, st_features_lstm2_ad, st_features_lstm1_ad = self.ad_encoder(x_ad)

        # Concat latent spaces of each view
        rd_ad_ra_latent_space = torch.cat((st_features_backbone_rd,
                                           st_features_backbone_ra,
                                           st_features_backbone_ad), dim=1)


        # Latent space for skip connection 2 - Range Doppler and Range Angle view (h_kskip_1 in the paper)
        # Concatenate
        latent_rd_ad_ra_skip_connection_2 = torch.cat((st_features_lstm2_rd,
                                                       st_features_lstm2_ra,
                                                       st_features_lstm2_ad), dim=1)
        # Reduce # channels
        latent_rd_ad_ra_skip_connection_2 = self.skip_connection_lstm2_conv(latent_rd_ad_ra_skip_connection_2)

        # Latent space for skip connection 1 (h_kskip_0 in the paper)
        # Skip connection for RD decoder - Down sample features map from RA view to match sizes of AD and RD views
        latent_skip_connection1_rd = torch.cat((st_features_lstm1_rd,
                                                self.down_sample_ra_view_skip_connection1(st_features_lstm1_ra),
                                                st_features_lstm1_ad), dim=1)
        latent_skip_connection1_rd = self.skip_connection_lstm1_conv(latent_skip_connection1_rd)

        # Skip connection for RA decoder - Up sample features maps from RD and AD view to match sizes of RA view
        latent_skip_connection1_ra = torch.cat((self.up_sample_rd_ad_views_skip_connection1(st_features_lstm1_rd),
                                  st_features_lstm1_ra,
                                  self.up_sample_rd_ad_views_skip_connection1(st_features_lstm1_ad)), dim=1)
        latent_skip_connection1_ra = self.skip_connection_lstm1_conv(latent_skip_connection1_ra)

        # Decode
        pred_rd = self.rd_decoder(rd_ad_ra_latent_space, latent_rd_ad_ra_skip_connection_2, latent_skip_connection1_rd)
        pred_ra = self.ra_decoder(rd_ad_ra_latent_space, latent_rd_ad_ra_skip_connection_2, latent_skip_connection1_ra)

        return pred_rd, pred_ra
