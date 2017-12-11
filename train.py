import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms

from model import MNISTModel


def train(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root='data', train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=256,
        sampler=sampler.SubsetRandomSampler(list(range(0, 55000))))
    valid_loader = DataLoader(
        dataset=train_dataset, batch_size=256,
        sampler=sampler.SubsetRandomSampler(list(range(55000, 60000))))
    test_loader = DataLoader(dataset=test_dataset, batch_size=256)

    model = MNISTModel()
    if args.gpu > -1:
        model.cuda(args.gpu)

    if args.loss_type == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss type')

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=0.0005)
        min_lr = 0.001
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)
        min_lr = 0.00001
    else:
        raise ValueError('Unknown optimizer')

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.1, patience=5, verbose=True,
        min_lr=min_lr)

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

    def var(tensor, volatile=False):
        if args.gpu > -1:
            tensor = tensor.cuda(args.gpu)
        return Variable(tensor, volatile=volatile)

    global_step = 0

    def train_epoch():
        nonlocal global_step
        model.train()
        for train_batch in train_loader:
            train_x, train_y = var(train_batch[0]), var(train_batch[1])
            logit = model(train_x)
            loss = criterion(input=logit, target=train_y)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()
            global_step += 1
            summary_writer.add_scalar(
                tag='train_loss', scalar_value=loss.data[0],
                global_step=global_step)

    def validate():
        model.eval()
        loss_sum = loss_denom = 0
        for valid_batch in valid_loader:
            valid_x, valid_y = (var(valid_batch[0], volatile=True),
                                var(valid_batch[1], volatile=True))
            logit = model(valid_x)
            loss = criterion(input=logit, target=valid_y)
            loss_sum += loss.data[0] * valid_x.size(0)
            loss_denom += valid_x.size(0)
        loss = loss_sum / loss_denom
        summary_writer.add_scalar(tag='valid_loss', scalar_value=loss,
                                  global_step=global_step)
        lr_scheduler.step(loss)
        return loss

    def test():
        model.eval()
        num_correct = denom = 0
        for test_batch in test_loader:
            test_x, test_y = (var(test_batch[0], volatile=True),
                                var(test_batch[1], volatile=True))
            logit = model(test_x)
            y_pred = logit.max(1)[1]
            num_correct += y_pred.eq(test_y).long().sum().data[0]
            denom += test_x.size(0)
        accuracy = num_correct / denom
        summary_writer.add_scalar(tag='test_accuracy', scalar_value=accuracy,
                                  global_step=global_step)
        return accuracy

    best_valid_loss = 1e10
    for epoch in range(1, args.max_epoch + 1):
        train_epoch()
        valid_loss = validate()
        print(f'Epoch {epoch}: Valid loss = {valid_loss:.5f}')
        test_accuracy = test()
        print(f'Epoch {epoch}: Test accuracy = {test_accuracy:.5f}')
        if valid_loss < best_valid_loss:
            model_filename = (f'{epoch:02d}'
                              f'-{valid_loss:.5f}'
                              f'-{test_accuracy:.5f}.pt')
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f'Epoch {epoch}: Saved the new best model to: {model_path}')
            best_valid_loss = valid_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-type', required=True)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--max-epoch', default=50, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save-dir', required=True)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
