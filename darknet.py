import torch
import torch.nn as nn

from layer import *

class LayerOne(BasicLayer):
    def __init__(self):
        super().__init__((64, 32, 1, 1, 0),
                         (32, 64, 3, 1, 1), 1)


class LayerTwo(BasicLayer):
    def __init__(self):
        super().__init__((128, 64, 1, 1, 0),
                         (64, 128, 3, 1, 1), 2)


class LayerThree(BasicLayer):
    def __init__(self):
        super().__init__((256, 128, 1, 1, 0),
                         (128, 256, 3, 1, 1), 8)


class LayerFour(BasicLayer):
    def __init__(self):
        super().__init__((512, 256, 1, 1, 0),
                         (256, 512, 3, 1, 1), 8)


class LayerFive(BasicLayer):
    def __init__(self):
        super().__init__((1024, 512, 1, 1, 0),
                         (512, 1024, 3, 1, 1), 4)


class FirstPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 route_index=4,
                 anchors=[(116, 90), (156, 198), (373, 326)]):
        super().__init__(structs, use_cuda, anchors, route_index=route_index)


class SecondPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 route_index=4,
                 anchors=[(30, 61), (62, 45), (59, 119)]):
        super().__init__(structs, use_cuda, anchors, route_index=route_index)


class ThirdPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes=80,
                 height=416,
                 anchors=[(10, 13), (16, 30), (33, 23)]):
        super().__init__(structs, use_cuda, anchors)


class DarkNet(nn.Module):
    def __init__(self, num_classes, use_cuda):
        super().__init__()

        self.num_classes = num_classes
        DETECT_DICT = {
            #         indim, BasicConv(out_dim, kr_size, stride, padding)                      # route layer
            'first':  [ 1024, 
                       ( 512, 1, 1, 0), 
                       (1024, 3, 1, 1), 
                       ( 512, 1, 1, 0), 
                       (1024, 3, 1, 1), 
                       ( 512, 1, 1, 0), 
                       (1024, 3, 1, 1), 
                       ( 255, 1, 1, 0, 0)],
            'second': [  768,  
                       ( 256, 1, 1, 0), 
                       ( 512, 3, 1, 1), 
                       ( 256, 1, 1, 0), 
                       ( 512, 3, 1, 1), 
                       ( 256, 1, 1, 0), 
                       ( 512, 3, 1, 1), 
                       ( 255, 1, 1, 0, 0)],
            'third':  [  384,  
                       ( 128, 1, 1, 0), 
                       ( 256, 3, 1, 1), 
                       ( 128, 1, 1, 0), 
                       ( 256, 3, 1, 1), 
                       ( 128, 1, 1, 0), 
                       ( 256, 3, 1, 1), 
                       ( 255, 1, 1, 0, 0)],
        }
        self.seq_1 = nn.Sequential(
            BasicConv(3, 32, 3, 1, 1),
            BasicConv(32, 64, 3, 2, 1),
            LayerOne(),
            BasicConv(64, 128, 3, 2, 1),
            LayerTwo(),
            BasicConv(128, 256, 3, 2, 1),
            LayerThree(),
        )

        self.conv_1 = BasicConv(256, 512, 3, 2, 1)
        
        self.layer_4 = LayerFour()

        self.seq_2 = nn.Sequential(
            BasicConv(512, 1024, 3, 2, 1),
            LayerFive(),
            FirstPred(DETECT_DICT["first"], use_cuda)
        )

        self.uns_1 = nn.Sequential(
            BasicConv(512, 256, 1, 1, 0),
            # nn.Upsample(scale_factor=2, mode="bilinear")
            # nn.functional.interpolate(..., scale_factor=2, mode="bilinear")
            BasicInterpolate(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.uns_2 = nn.Sequential(
            BasicConv(256, 128, 1, 1, 0),
            # nn.Upsample(scale_factor=2, mode="bilinear")
            # nn.functional.interpolate(..., scale_factor=2, mode="bilinear")
            BasicInterpolate(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.pred_2 = SecondPred(DETECT_DICT["second"], use_cuda)
        self.pred_3 = ThirdPred(DETECT_DICT["third"], use_cuda)

    def forward(self, x):
        x = self.seq_1(x)
        r_0 = x # 256 52x52 1/8*1/8

        x = self.layer_4(self.conv_1(x))
        r_1 = x # 512 26x26 1/16*1/16

        det_1, x = self.seq_2(x) # det_1: 13*13*3 85  ,x: 512 13x13 1/32*1/32

        x = self.uns_1(x)
        x = torch.cat((x, r_1), 1) # cat: 256 26*26 + 512 26*26 = 768 26*26
        det_2, x = self.pred_2(x) #

        x = self.uns_2(x)
        x = torch.cat((x, r_0), 1)
        det_3 = self.pred_3(x)

        return torch.cat((det_1, det_2, det_3), 1)
