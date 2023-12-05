import torch
import torch.nn as nn
import torch.nn.functional as F
from convLSTM import ConvLSTMCell, ConvLSTM

# Define the full model that combines the encoder, ConvLSTM-based reconstructor, and decoder
class VideoFrameReconstructor(nn.Module):
    def __init__(self, num_frames, hidden_size, num_layers, C, H, W):
        super(VideoFrameReconstructor, self).__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder and Decoder should have appropriate parameters passed
        self.encoder = Encoder(C, hidden_size)  # Assuming C is the number of channels
        self.reconstructor = ConvLSTM(C, hidden_size, kernel_size=(3,3), num_layers=num_layers, batch_first=True)
        # self.decoder = Decoder(hidden_size, C)  # Assuming the output has the same channels as input

    def forward(self, x):
        batch_size, _, _, H, W = x.size()  # x is expected to be of shape [B, T, C, H, W]

        # Encode each frame
        encoded_frames = [self.encoder(x[:, t, :, :, :]) for t in range(self.num_frames)]

        # check the size of encoded_frames
        print("Dimension of encoded_frames:", encoded_frames[0].shape)

        # Stack encoded frames and pass through ConvLSTM
        encoded_sequence = torch.stack(encoded_frames, dim=1)  # Shape [B, T, C, H', W']
        _, last_states = self.reconstructor(encoded_sequence)

        # Use the final hidden state to predict the next frame
        final_hidden = last_states[-1][0]  # Assuming return_all_layers=False

        # check the size of final_hidden
        print("Dimension of final_hidden:", final_hidden.shape)

        return final_hidden
        
        # next_frame_prediction = self.decoder(final_hidden)

        # return next_frame_prediction
    

# Define the encoder structure
class Encoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # check the size of encoded_frames
        print("Dimension of each encoded frame:", x5.shape)

        return x5


# Define the decoder structure
class Decoder(nn.Module):
    def __init__(self, n_classes, size_rerorder, bilinear=False):
        super(Decoder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.x1_size = size_rerorder[0]
        self.x2_size = size_rerorder[1]
        self.x3_size = size_rerorder[2]
        self.x4_size = size_rerorder[3]

    def forward(self, x):
        x = self.up1(x, self.x4_size)
        x = self.up2(x, self.x3_size)
        x = self.up3(x, self.x2_size)
        x = self.up4(x, self.x1_size)
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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2_size):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2_size[2] - x1.size()[2]
        diffX = x2_size[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)