import torch
from torch import nn

class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, relu=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.relu = relu

        model = [
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride),
            nn.BatchNorm2d(self.out_channels),
        ]
        if self.relu:
            model.append(nn.ReLU())
        self.model = nn.Sequential(*model)
    
    def forward(self, X):
        return self.model(X)
    
class Residual_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.model = nn.Sequential(
            MyConv(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=stride, relu=True),
            MyConv(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1, relu=False),
        )

        if self.in_channels != self.out_channels or stride != 1:
            self.downsample = nn.Sequential(
                MyConv(self.in_channels, self.out_channels, kernel_size=1, padding=0, stride=stride, relu=False),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, X):
        X_skip = X
        X_skip = self.downsample(X_skip)
        out = self.model(X)
        out = out + X_skip
        return out

class Residual_Bottle_Neck_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.model = nn.Sequential(
            MyConv(self.in_channels, self.out_channels // 4, kernel_size=1, padding=0, stride=stride, relu=True),
            MyConv(self.out_channels // 4, self.out_channels // 4, kernel_size=3, padding=1, stride=1, relu=True),
            MyConv(self.out_channels // 4, self.out_channels, kernel_size=1, padding=0, stride=1, relu=False),
        )

        if self.in_channels != self.out_channels or stride != 1:
            self.downsample = nn.Sequential(
                MyConv(self.in_channels, self.out_channels, kernel_size=1, padding=0, stride=stride, relu=False),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, X):
        X_skip = X
        X_skip = self.downsample(X_skip)
        out = self.model(X)
        out = out + X_skip
        return out
    
def make_block(block, in_channels, out_channels, num_blocks, stride):
    model = nn.Sequential(
        block(in_channels, out_channels, stride),
        *[block(out_channels, out_channels, 1) for _ in range(num_blocks-1)]
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

        model = []
        for idx, num_blocks in enumerate(list_num_residual_blocks):
            if idx == 0:
                model.append(make_block(self.block, 64, 64*expansion_size, num_blocks, stride=2))
            else:
                model.append(make_block(self.block, 64*expansion_size*2**(idx-1), 64*expansion_size*2**idx, num_blocks, stride=2))
        self.model = nn.Sequential(*model)

        self.averagepool = nn.AvgPool2d(2)
        self.fc = nn.Linear((in_w//2**6)*(in_h//2**6)*512*expansion_size, num_classes)
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.model(X)
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
