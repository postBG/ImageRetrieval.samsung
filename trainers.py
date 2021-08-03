import torch
import torch.nn as nn
from tqdm import tqdm

from utils import AverageMeterSet


def train_one_epoch(model, optimizer, trainloader):
    criterion = nn.CrossEntropyLoss()
    trainloader = tqdm(trainloader)
    average_meter_set = AverageMeterSet()

    model.train()
    for i, (xs, ys, _) in enumerate(trainloader):
        xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(xs)
        loss = criterion(logits, ys)
        loss.backward()
        average_meter_set.update('loss', loss.item())
        optimizer.step()

    return model


@torch.no_grad()
def evaluation(model, valloader, epoch):
    criterion = nn.CrossEntropyLoss()
    average_meter_set = AverageMeterSet()
    valloader = tqdm(valloader)

    model.eval()
    for i, (xs, ys, _) in enumerate(valloader):
        xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)

        logits = model(xs)

        _, predictions = logits.max(1)
        average_meter_set.update('val_loss', criterion(logits, ys))
        average_meter_set.update('num_corrects', predictions.eq(ys).sum().item(), n=predictions.size(0))

    print("Epoch: {}, val_loss: {:.3f}, Accuracy: {:.3f}\n".format(
        epoch, average_meter_set['val_loss'].avg, 100 * average_meter_set['num_corrects'].avg,
    ))


def train(model, optimizer, trainloader, valloader, epoch):
    model.cuda()
    for e in range(epoch):
        model = train_one_epoch(model, optimizer, trainloader)
        evaluation(model, valloader, e)
    return model.cpu()
