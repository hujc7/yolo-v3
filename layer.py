import torch
import torch.nn as nn
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, ind, outd, kr_size, stride, padding, lr=0.1, bias=False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(ind, outd, kr_size, stride, padding, bias=bias),
            nn.BatchNorm2d(outd),
            nn.LeakyReLU(lr, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class BasicLayer(nn.Module):
    def __init__(self, conv_1, conv_2, times):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(times):
            self.layers.append(BasicConv(*conv_1))
            self.layers.append(BasicConv(*conv_2))

    def forward(self, x):
        residual = x
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index % 2 == 1:
                x += residual
                residual = x

        return x

class BasicInterpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()

        self._size = size
        self._scale_factor = scale_factor
        self._mode = mode
        self._align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, size=self._size, scale_factor=self._scale_factor, mode=self._mode, align_corners = self._align_corners)

class BasicPred(nn.Module):
    def __init__(self,
                 structs,
                 use_cuda,
                 anchors,
                 classes=80,
                 height=416,
                 route_index=0 # the index of layer for output
                 ):
        super().__init__()

        self.ri = route_index
        self.classes = classes
        self.height = height
        self.anchors = anchors
        self.torch = torch.cuda if use_cuda else torch

        in_dim = structs[0]
        self.layers = nn.ModuleList()
        for s in structs[1:]:
            if len(s) == 4:
                out_dim, kr_size, stride, padding = s
                layer = BasicConv(in_dim, out_dim, kr_size, stride, padding)
            else: 
                # fifth elemnt indicate not a BasicConv but Conv2d
                # last layer for prediction is dimension reduction
                out_dim, kr_size, stride, padding, _ = s
                layer = nn.Conv2d(in_dim, out_dim, kr_size, stride, padding)

            in_dim = out_dim
            self.layers.append(layer)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.ri != 0 and index == self.ri: # return the output of ri layer as output
                output = x

        detections = self.predict_transform(x.data)

        if self.ri != 0:
            return detections, output
        else:
            return detections

    def predict_transform(self, prediction):
        """ borrowed from https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/util.py#L47
        """
        # prediction = batch_size*num_channels*height*width
        batch_size = prediction.size(0)
        stride = self.height // prediction.size(2) # 32, 16, 8: pixels per cell
        grid_size = self.height // stride          # 13, 26, 52: number of cells
        bbox_attrs = 5 + self.classes              # number of predictions per cell
        num_anchors = len(self.anchors)            # number of anchors

        prediction = prediction.view(
            batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(
            batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors] # normalized to [0, grid_size]

        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0]) # tx
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1]) # ty

        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = self.torch.FloatTensor(a).view(-1, 1) # column vector: grid_size * 1
        y_offset = self.torch.FloatTensor(b).view(-1, 1) # column vector: grid_size * 1

        # grid_size * 2
        # grid_size * (2*num_anchors)
        # (grid_size*num_anchors) * 2
        x_y_offset = torch.cat((x_offset, y_offset), 1) \
                          .repeat(1, num_anchors) \
                          .view(-1, 2) \
                          .unsqueeze(0)

        prediction[:, :, :2] += x_y_offset

        anchors = self.torch.FloatTensor(anchors)

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors # range [0, grid_size]
        prediction[:, :, :4] *= stride # convert to original size 416*416

        # sigmoid Objectness and classes confidence
        prediction[:, :, 4:] = torch.sigmoid(prediction[:, :, 4:])

        return prediction
