import abc
import sys
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torch import autograd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import math


from torchvision.transforms import transforms


# Meter for logging
class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


# AverageMeter for logging
class AverageMeter(Meter):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MaskedCeil(autograd.Function):
    @staticmethod
    def forward(ctx, mask, prune_rate):
        output = mask.clone()
        _, idx = mask.flatten().abs().sort()
        p = int(prune_rate * mask.numel())

        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


# For learning the mask
class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskConv, self).__init__(*args, **kwargs)

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))

        self.weight.data = self.weight.data.sign() * 0.1

        self.weight.requires_grad = False
        self.prune_rate = 0.7

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_mask(self):
        return self.mask.abs()

    def forward(self, x):
        mask = MaskedCeil.apply(self.clamped_mask, self.prune_rate)

        w = self.weight * mask
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return MaskConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.linear = nn.Sequential(
            conv1x1(28 * 28, 300),
            nn.ReLU(),
            conv1x1(300, 100),
            nn.ReLU(),
            conv1x1(100, 10),
        )

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()


# data
data_root = sys.argv[1]

use_cuda = torch.cuda.is_available()

# Data loading code
kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=128,
    shuffle=True,
    **kwargs
)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        data_root,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=128,
    shuffle=True,
    **kwargs
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FC()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    momentum=0.9,
    lr=0.1,
    weight_decay=1e-4,
    nesterov=False,
)

epoch_len = len(train_loader)
num_epochs = 90
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, epoch_len * num_epochs
)

acc = AverageMeter("Accuracy", ":.3f")
for epoch in range(num_epochs):

    ## train
    model.train()
    for idx, (data, label) in enumerate(train_loader):
        logits = model(data.to(device))
        loss = criterion(logits, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (idx % 100) == 0:
            print(
                "Epoch: [{}][{}/{}]\tLoss {:.3f}\t".format(
                    epoch, idx, len(train_loader), loss.item()
                )
            )

    # val
    acc.reset()
    model.eval()
    for idx, (data, label) in enumerate(val_loader):
        logits = model(data.to(device))
        acc1, _ = accuracy(logits, label.to(device), topk=(1, 5))
        acc.update(acc1.item(), logits.size(0))
    print("VAL EPOCH {} : {:.3f}".format(epoch, acc.avg))
