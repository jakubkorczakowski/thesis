from torch import nn
import torch


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(
                pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            #IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b


class DSLoss_IoU_noCAM(nn.Module):
    def __init__(self):
        super(DSLoss_IoU_noCAM, self).__init__()
        self.iou = IoU_loss()

    def forward(self, scaled_preds, gt):
        [_, num, _, _, _] = gt.size()
        loss = 0
        for inum in range(num):
            for isout, x in enumerate(scaled_preds[inum][1:]):
                loss += self.iou(x, gt[:, inum, :, :, :])
        return loss
