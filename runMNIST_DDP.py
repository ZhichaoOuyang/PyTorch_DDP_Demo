from __future__ import print_function
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from util.DDPUtil import set_DDP_device, master_print, DDP_prepare, move_model_to_device, move_to_device, \
    use_multi_GPUs, is_master, get_device


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(log_interval, model, train_loader, train_data_sampler, optimizer, epoch):
    model.train()

    # DDP Step 4: Manually shuffle to avoid a known bug for DistributedSampler.
    # https://github.com/pytorch/pytorch/issues/31232
    # https://github.com/pytorch/pytorch/issues/31771
    if use_multi_GPUs():
        train_data_sampler.set_epoch(epoch)

    # DDP Step 5: Only record the global loss value and other information in the master GPU.
    if is_master():
        global_cumulative_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = move_to_device(data), move_to_device(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        # DDP Step 6: Collect loss value and other information from each GPU to the master GPU.
        if use_multi_GPUs():
            # If more information is needed, add to this tensor.
            result = torch.tensor([loss_value], device=get_device())

            dist.barrier()
            # Get the sum of results from all GPUs
            dist.all_reduce(result, op=dist.ReduceOp.SUM)

            # Only master GPU records all the information
            if is_master():
                result = result.tolist()
                global_cumulative_loss += result[0]
        else:
            # use single GPU or CPU
            global_cumulative_loss += loss_value

        if is_master() and batch_idx % log_interval == 0:
            master_print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), global_cumulative_loss))
            # if args.dry_run:
            #     break


def test(model, test_loader):
    model.eval()
    # test_loss = 0
    # correct = 0

    # DDP Step 5: Only record the global loss value and other information in the master GPU.
    if is_master():
        global_cumulative_loss = 0
        global_correct = 0

    with torch.no_grad():

        for data, target in test_loader:
            data, target = move_to_device(data), move_to_device(target)
            output = model(data)
            test_loss_value = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()


            # DDP Step 6: Collect loss value and other information from each GPU to the master GPU.
            if use_multi_GPUs():
                # If more information is needed, add to this tensor.
                result = torch.tensor([test_loss_value, correct], device=get_device())

                dist.barrier()
                # Get the sum of results from all GPUs
                dist.all_reduce(result, op=dist.ReduceOp.SUM)

                # Only master GPU records all the information
                if is_master():
                    result = result.tolist()
                    global_cumulative_loss += result[0]
                    global_correct += result[1]
            else:
                # use single GPU or CPU
                global_cumulative_loss += test_loss_value
                global_correct += correct

    if is_master():
        global_cumulative_loss /= len(test_loader.dataset)
        master_print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            global_cumulative_loss, global_correct, len(test_loader.dataset),
            100. * global_correct / len(test_loader.dataset)))


def main():
    # Training settings
    batch_size = 64
    epochs = 14
    lr = 1.0
    gamma = 0.7
    log_interval = 10
    save_model = False
    # Number of processes for dataloader (work in CPU)
    num_workers = 1

    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=14, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # DDP Step 1: Devices and random seed are set in set_DDP_device().
    # torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")


    # kwargs = {'batch_size': args.batch_size}
    # if use_cuda:
    #     kwargs.update({'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True},
    #                  )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    # DDP Step 2: Move model to devices.
    model = Net()
    move_model_to_device(model)

    # model = Net().to(device)

    # DDP Step 3: Use DDP_prepare to prepare datasets and loaders.
    model, train_loader, test_loader, train_data_sampler, test_data_sampler = DDP_prepare(
        train_dataset=dataset1,
        test_dataset=dataset2,
        num_data_processes=num_workers,
        global_batch_size=batch_size,
        # In case you have sophisticated data processing function, pass it to collate_fn (i.e., collate_fn of the DataLoader)
        collate_fn=None, model=model)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train(log_interval, model, train_loader, train_data_sampler, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

    end = time.perf_counter()
    master_print("Total Training Time %.2f seconds" % (end - start))

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':

    # DDP Step 1: Set devices and the random seed for reproducibility
    # gpu_id = "-1"     # use CPU
    # gpu_id = "0"      # use single GPU 0
    gpu_id = "0,1,2,3"  # use GPUs 0, 1, 2, 3

    set_DDP_device(gpu_id)

    main()

