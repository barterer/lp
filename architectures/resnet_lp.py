import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        
        out = self.conv1(x)
        out = self.quant(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quant(out)

        
        out = self.conv2(out)
        out = self.quant(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.quant(residual)

        return self.quant(out)
    
class LPResNet_baby(nn.Module):

    def __init__(self,low_quant,high_quant,num_blocks, num_classes=10):

        super(LPResNet_baby, self).__init__()
        
        block = BasicBlock

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], low_quant, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], low_quant, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], low_quant, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], low_quant, stride=2)
        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.low_quant = low_quant()
        self.high_quant = high_quant
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.high_quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.low_quant(x)

        x = self.layer1(x) 
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
        # x = self.bn(x)
        # x = self.relu(x)
        # x = self.low_quant(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.high_quant(x)

        return x

class LPResNet(nn.Module):

    def __init__(self,low_quant,high_quant,num_blocks, num_classes=10):

        super(LPResNet, self).__init__()
        
        block = BasicBlock

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], low_quant, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], low_quant, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], low_quant, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], low_quant, stride=2)
        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.low_quant = low_quant()
        self.high_quant = high_quant
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.high_quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.low_quant(x)

        x = self.layer1(x) 
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
        # x = self.bn(x)
        # x = self.relu(x)
        # x = self.low_quant(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.high_quant(x)

        return x

def ResNet18(low_quant_func, high_quant, num_classes):
    return LPResNet_baby(low_quant_func, high_quant, [2, 2, 2, 2], num_classes)