import torch
import torch.nn as nn
import torch.nn.functional as F
from .convLSTM import ConvLSTMCell, ConvLSTM

# Define the full model that combines the encoder, ConvLSTM-based reconstructor, and decoder
class VideoFrameReconstructor_Mini(nn.Module):
    def __init__(self, C, num_frames, LSTM_hidden_size, num_layers, encoder_final_channel = 512):
        super(VideoFrameReconstructor_Mini, self).__init__()
        self.num_frames = num_frames
        self.hidden_size = LSTM_hidden_size
        self.num_layers = num_layers

        # Encoder and Decoder should have appropriate parameters passed
        self.encoder = Encoder(input_channels=C, final_channel=encoder_final_channel)
        self.reconstructor = ConvLSTM(input_dim=encoder_final_channel, hidden_dim=LSTM_hidden_size, kernel_size=(3, 3), num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.decoder = Decoder(input_channels=LSTM_hidden_size, final_channel=C)

    def forward(self, x):
        batch_size, _, _, H, W = x.size()  # x is expected to be of shape [B, T, C, H, W]

        # Encode each frame
        encoded_frames = [self.encoder(x[:, t, :, :, :]) for t in range(self.num_frames)]

        # Stack encoded frames and pass through ConvLSTM
        encoded_sequence = torch.stack(encoded_frames, dim=1)  # Shape [B, T, C, H', W']

        _, last_states = self.reconstructor(encoded_sequence)

        # Use the final hidden state to predict the next frame
        final_hidden = last_states[-1][0]  # Assuming return_all_layers=False
        
        next_frame_prediction = self.decoder(final_hidden)

        return next_frame_prediction
    

# Define the encoder structure
class Encoder(nn.Module):
    def __init__(self, input_channels, final_channel, bilinear=False):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(input_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down3 = (Down(256, final_channel // factor))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return x4


# Define the decoder structure
class Decoder(nn.Module):
    def __init__(self, input_channels = 256, final_channel = 3, bilinear=False):
        super(Decoder, self).__init__()
        # Adjust the initial_channels to match the output channels of ConvLSTM
        factor = 2 if bilinear else 1
        self.up1 = Up(input_channels, 128 // factor, bilinear)
        self.up2 = Up(128 // factor, 64, bilinear)
        self.up3 = Up(64 // factor, 32, bilinear)
        self.outc = OutConv(32, final_channel)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv without skip connections"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Adjust the DoubleConv to only consider in_channels
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)