import torch, os
from torch import nn, Tensor
import numpy as np

from typing import Optional
from torch import Tensor
from typing import List, Type, Union

def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation, stride=stride, padding=dilation, bias=False)

def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(
            planes,
            planes,
            stride=stride,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.layers = layers
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(
        self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1, dilation: int = 1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride=stride,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    

class MedicalNet_L1Layers(torch.nn.Module):
    def __init__(self, opt=None, freeze_weights = True):
        super().__init__()
        self.opt = opt
        # resnet_pretrained = ResNet50(BasicBlock, [1, 1, 1, 1]) #resnet10
        resnet_pretrained = ResNet50(Bottleneck, [3, 4, 6, 3]) #resnet50
        resnet_pretrained = resnet_pretrained.to(device='cuda', dtype=torch.float16) 

        checkpoint = torch.load("/data/alonguefosse/checkpoints/resnet_50_23dataset.pth") #cpu ?
        pretrained_state_dict = checkpoint["state_dict"]
        pretrained_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}

        resnet_pretrained.load_state_dict(pretrained_state_dict, strict=False)

        resnet_pretrained.eval() 
        
        self.conv1 = resnet_pretrained.conv1
        self.bn1 = resnet_pretrained.bn1
        self.relu = resnet_pretrained.relu
        self.maxpool = resnet_pretrained.maxpool
        self.layer1 = resnet_pretrained.layer1
        self.layer2 = resnet_pretrained.layer2
        self.layer3 = resnet_pretrained.layer3
        self.layer4 = resnet_pretrained.layer4
        for param in self.parameters():
            param.requires_grad = False
        
        self.L1 = nn.L1Loss()
        self.debug = False

    def forward(self, x, y):
        h = self.conv1(x)
        h = self.bn1(h)
        h_relu1 = self.relu(h)
        h = self.maxpool(h_relu1)
        h_conv1 = self.layer1(h)
        h_conv2 = self.layer2(h_conv1)
        h_conv3 = self.layer3(h_conv2)
        h_conv4 = self.layer4(h_conv3)

        emb_x = [h_relu1, h_conv1, h_conv2, h_conv3, h_conv4]

        h2 = self.conv1(y)
        h2 = self.bn1(h2)
        h2_relu1 = self.relu(h2)
        h2 = self.maxpool(h2_relu1)
        h2_conv1 = self.layer1(h2)
        h2_conv2 = self.layer2(h2_conv1)
        h2_conv3 = self.layer3(h2_conv2)
        h2_conv4 = self.layer4(h2_conv3)

        emb_y = [h2_relu1, h2_conv1, h2_conv2, h2_conv3, h2_conv4]

        sum_loss = 0
        layers_losses = []
        for i in range(len(emb_y)):
            layer_loss = self.L1(emb_x[i], emb_y[i].detach())
            sum_loss += layer_loss
            layers_losses.append(layer_loss)
    
            if self.debug:
                print("mednet", i, emb_x[i].shape, layer_loss)

        # print(sum_loss)

        with open(f'mednet.txt', 'a') as file:
            file.write(f"x = {torch.min(x)}, {torch.max(x)}\n")
            file.write(f"y = {torch.min(y)}, {torch.max(y)}\n")
            for i in range(len(layers_losses)):
                file.write(f"Layer {i}: Loss = {layers_losses[i]}\n")
            file.write(f"-------------------\n")

        

        return sum_loss