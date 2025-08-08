import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class SmallCNN(nn.Module):
    """
    3-stage CNN. 중간 stage 출력 뒤 SuppressiveDropout을 '한 번만' 적용.
    """
    def __init__(self, num_classes=10, drop_layer=None):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64), nn.MaxPool2d(2)  # 32->16
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2)  # 16->8
        )
        self.sdrop = drop_layer  # 여기에만 적용
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256), ConvBlock(256, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        if self.sdrop is not None:
            x = self.sdrop(x)      # 🔴 단 한 번
        x = self.stage3(x)
        x = self.head(x)
        return x

def get_cnn(num_classes=10, drop_layer=None):
    return SmallCNN(num_classes=num_classes, drop_layer=drop_layer)
