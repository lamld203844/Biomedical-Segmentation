import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = F.one_hot(target, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(output * target, dim=(2, 3))
        fp = torch.sum(output * (1 - target), dim=(2, 3))
        fn = torch.sum((1 - output) * target, dim=(2, 3))
        
        tversky = (intersection + self.eps) / (intersection + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1. - tversky
        return loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = F.one_hot(target, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()
        
        pt = torch.where(target == 1, output, 1 - output)
        logpt = torch.log(pt + self.eps)
        
        loss = -1 * self.alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean()

class IoULoss(nn.Module):

    def __init__(self, epsilon=1e-6):
        super(IoULoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1) # Apply a softmax function to the prediction logits -> labels
        y_pred = y_pred[:, 1, :, :] # Reshape the tensor to [8, 128, 128]

        # Flatten the tensors
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        # Compute the intersection and the union
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred) - intersection

        # Compute the IoU coefficient and return the loss
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return -torch.log(iou)
    
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = F.one_hot(target, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(output * target, dim=(2, 3))
        union = torch.sum(output, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        
        dice = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1. - dice
        return loss.mean()
    
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = targets.squeeze(1)
        targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets = targets.permute(0, 3, 1, 2).float()
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.shape[1])
        targets = targets.permute(0, 2, 3, 1).contiguous().view(-1, targets.shape[1])
        losses = []
        for i in range(inputs.shape[1]):
            target = targets[:, i]
            input = inputs[:, i]
            loss = (Variable(target) - input).abs()
            loss_sorted, loss_sorted_ind = torch.sort(loss, 0, descending=True)
            target_sorted = target[loss_sorted_ind]
            losses.append(torch.dot(loss_sorted, Variable(lovasz_grad(target_sorted))))
        losses = torch.stack(losses)
        return losses.mean()