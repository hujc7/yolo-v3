# -*- coding: utf-8 -*-
import torch
import torch.nn as nnimport torch.nn.functional as F



class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes, overlap_thresh, bkg_label, 
                 neg_pos_traio, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.bkg_label = bkg_label
        self.neg_pos_ratio = neg_pos_traio
        self.use_gpu = use_gpu

    def forward(self, predictions, targets):
        
        pass