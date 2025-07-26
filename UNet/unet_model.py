import torch
import torch.nn as nn

# --- U-Net Building Blocks ---

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    This is the standard convolutional block used throughout the U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Using ConvTranspose2d for learned upsampling.
        # It's generally preferred over simple interpolation for better performance.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels) # in_channels accounts for skip connection concatenation

    def forward(self, x1, x2):
        """
        x1 is the feature map from the previous decoder stage (upsampled).
        x2 is the corresponding feature map from the encoder (skip connection).
        """
        x1 = self.up(x1)
        # Pad x1 if its size is slightly different from x2 due to padding/stride in encoder
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # Concatenate along the channel dimension
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution to map channels to num_classes
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# --- Standard U-Net Model ---

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (Downsampling Path)
        # Each 'down' block consists of MaxPool and then DoubleConv
        self.inc = DoubleConv(in_channels, 64)  # Initial convolution
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Bottleneck layer

        # Decoder (Upsampling Path)
        # Each 'up' block consists of ConvTranspose2d (upsample) and then DoubleConv
        # Note the input channels for 'Up' are (features from previous stage + skip connection features)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Output convolution
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # Output: 64 channels, same size as input
        x2 = self.down1(x1)      # Output: 128 channels, 1/2 size
        x3 = self.down2(x2)      # Output: 256 channels, 1/4 size
        x4 = self.down3(x3)      # Output: 512 channels, 1/8 size
        x5 = self.down4(x4)      # Output: 1024 channels, 1/16 size (bottleneck)

        # Decoder with Skip Connections
        # up(upsampled feature, skip connection feature)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        logits = self.outc(x)
        return logits
