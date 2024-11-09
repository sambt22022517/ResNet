import torch
from torch import nn

class Residual_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
        )

        if self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, X):
        skip_connect_X = X.clone()
        downsample_X = self.downsample(skip_connect_X)

        X = self.model(X)

        X = X + downsample_X
        return X
    
class Residual_Bottle_Neck_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(self.out_channels // 4),

            nn.Conv2d(self.out_channels // 4, self.out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels // 4),

            nn.Conv2d(self.out_channels // 4, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )

        if self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, X):
        skip_connect_X = X.clone()
        downsample_X = self.downsample(skip_connect_X)

        X = self.model(X)

        X = X + downsample_X
        return X
    
def make_block(block, in_channels, out_channels, num_blocks):
    model = nn.Sequential(
        block(in_channels, out_channels),
        *[block(out_channels, out_channels) for _ in range(num_blocks-1)]
    )
    return model

class ResNet(nn.Module):
    def __init__(self, Residual_Block, in_channels, in_w, in_h, num_classes, list_num_residual_blocks, expansion_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.block = Residual_Block
        if expansion_size < 1:
            expansion_size = 1

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = make_block(self.block, 64, 64*expansion_size, list_num_residual_blocks[0])
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_x = make_block(self.block, 64*expansion_size, 128*expansion_size, list_num_residual_blocks[1])
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_x = make_block(self.block, 128*expansion_size, 256*expansion_size, list_num_residual_blocks[2])
        
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv5_x = make_block(self.block, 256*expansion_size, 512*expansion_size, list_num_residual_blocks[3])

        self.averagepool = nn.AvgPool2d(2)
        self.fc = nn.Linear((in_w//2**6)*(in_h//2**6)*512*expansion_size, num_classes)
    
    def forward(self, X):
        X = self.conv1(X)

        X = self.maxpool2(X)
        X = self.conv2_x(X)

        X = self.maxpool3(X)
        X = self.conv3_x(X)
        
        X = self.maxpool4(X)
        X = self.conv4_x(X)
        
        X = self.maxpool5(X)
        X = self.conv5_x(X)

        X = self.averagepool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.fc(X)
        return X
    
def ResNet18(in_channels, in_w, in_h, num_classes):
    return ResNet(Residual_Basic_Block, in_channels, in_w, in_h, num_classes, [2, 2, 2, 2], expansion_size=1)

def ResNet34(in_channels, in_w, in_h, num_classes):
    return ResNet(Residual_Basic_Block, in_channels, in_w, in_h, num_classes, [3, 4, 6, 3], expansion_size=1)

def ResNet50(in_channels, in_w, in_h, num_classes):
    return ResNet(Residual_Bottle_Neck_Block, in_channels, in_w, in_h, num_classes, [3, 4, 6, 3], expansion_size=4)

def ResNet101(in_channels, in_w, in_h, num_classes):
    return ResNet(Residual_Bottle_Neck_Block, in_channels, in_w, in_h, num_classes, [3, 4, 23, 3], expansion_size=4)

def ResNet152(in_channels, in_w, in_h, num_classes):
    return ResNet(Residual_Bottle_Neck_Block, in_channels, in_w, in_h, num_classes, [3, 8, 36, 3], expansion_size=4)
    
def test(resnet):
    a = torch.randn(2, 3, 224, 224)
    model = resnet(3, 224, 224, 100)
    print('Pass' if model(a).shape == torch.Size([2, 100]) else 'Fail')

if __name__ == '__main__':
    test(ResNet18)
    test(ResNet34)
    test(ResNet50)
    test(ResNet101)
    test(ResNet152)