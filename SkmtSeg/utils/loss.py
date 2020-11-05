import torch
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self, args,weight=None, size_average=True, batch_average=True, ignore_index=255):
        super(Loss,self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average

        #两个参数
        self.sigma = nn.Parameter(torch.rand([1]))
        self.beta =nn.Parameter(torch.rand([1]))
        self.args=args

        if(self.args.auxiliary is not None):
            self.loss_auxiliary=self.build_loss(mode='ce')
        self.loss_trunk=self.build_loss(mode='focal')

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        elif mode == 'focal':
            return nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        else:
            raise NotImplementedError

    def forward(self, preds,labels):
        loss2 = self.loss_trunk(preds['out'], labels['centre_labels'])
        if(self.args.auxiliary is not None):
            loss1=self.loss_auxiliary(preds['outer_out'],labels['outer_labels'])
            loss = (loss1 + loss2).mean()
        else:
            loss =loss2.mean()

        return loss



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




