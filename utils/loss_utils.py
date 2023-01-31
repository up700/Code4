# -*- coding:utf-8 -*-
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        
        return output, None

def get_sp_loss(input, target, temp):
    criterion = nn.NLLLoss(reduction='none').cuda()
    loss = torch.mul(criterion(torch.log(1 - F.softmax(input / temp, dim=1)), target.detach()), 1).mean()
    return loss

class GCELoss(nn.Module):

    def __init__(self, q=0.7, ignore_index=-100):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
             
    def forward(self, logits, targets, weights):
        # valid_idx = targets != self.ignore_index
        # logits = logits[valid_idx]
        # targets = targets[valid_idx]
        # weights = weights[valid_idx]
        # vanilla cross entropy when q = 0
        if self.q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
        else:
            if logits.size(-1) == 1:
                pred = torch.sigmoid(logits)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(logits, dim=-1)
            pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
            loss = (1-pred**self.q) / self.q
        loss = (loss.view(-1)*weights).sum() / weights.sum()
        return loss