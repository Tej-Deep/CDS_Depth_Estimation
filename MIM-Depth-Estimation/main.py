from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from dataset.base_dataset import get_dataset
from utils.criterion import SiLogLoss

    # torch.Size([8, 3, 448, 576])
    # torch.Size([8, 448, 576])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 1)
        self.conv2 = nn.Conv2d(32, 64, 8, 2)
        self.conv3 = nn.Conv2d(64, 24, 16, 4,)
        self.conv4 = nn.Conv2d(24, 16, 16, 4)
        self.conv5 = nn.Conv2d(16, 576, 1, 1)
        # self.conv4 = nn.Conv2d(448, 1, 1, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8640, 576)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print("x",x.size())
        
        x = self.conv1(x)
        #print("x",x.size())
        
        x = F.relu(x)
        #print("x",x.size())
        
        x = self.conv2(x)
        #print("x",x.size())
        
        x = F.relu(x)
        #print("x",x.size())
        
        x = F.max_pool2d(x, 2)
        #print("x",x.size())
        
        x = self.dropout1(x)
        #print("x",x.size())
        
        x = F.relu(x)
        #print("x",x.size())
        
        x = self.dropout2(x)
        #print("x",x.size())
        
        x = self.conv3(x)
        #print("x",x.size())
        
        x = F.relu(x)
        #print("x",x.size())
        
        x = self.conv4(x)
        #print("x",x.size())
        
        x = F.relu(x)
        #print("x",x.size())
        
        x = self.conv5(x)
        #print("x",x.size())
        
        x = torch.flatten(x, 1)
        #print("x",x.size())
        
        x = self.fc1(x)
        #print("x",x.size())
        
        # x = torch.squeeze(x, 1)
        # #print("x",x.size())
        
        # x = self.fc2(x)
        # #print("x",x.size())
        
        # x = self.conv4(x)
        # #print("x",x.size())
        
        # x = torch.squeeze(x, 1)
        # #print("x",x.size())
        
        # output = x
        output = F.log_softmax(x)
        print("output",output.size())

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # print(f'This is batch {batch_idx}')
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        # print(input_RGB.size())
        # print(depth_gt.size())

        input_RGB, depth_gt = input_RGB.to(device), depth_gt.to(device)
        optimizer.zero_grad()
        output = model(input_RGB)
        # print(output['pred_d'].squeeze())
        # print(depth_gt)
        # loss = SiLogLoss(output['pred_d'].squeeze(), depth_gt)
        # print(torch.max(depth_gt, 1)[1])
        loss = F.multilabel_margin_loss(output, torch.max(depth_gt, 1)[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_RGB), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_RGB = batch['image'].to(device)
            depth_gt = batch['depth'].to(device)
            output = model(input_RGB)
            test_loss += F.multilabel_margin_loss(output, torch.max(depth_gt, 1)[1], reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(torch.max(depth_gt, 1)[1].view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # train_kwargs = {'batch_size': args.batch_size}
    # test_kwargs = {'batch_size': args.test_batch_size}
    # if use_cuda:
    #     cuda_kwargs = {'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    
    
    dataset_kwargs = {'dataset_name': 'nyudepthv2', 'data_path': './data/'}
    # dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                                shuffle=True, num_workers=0, 
                                                pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                pin_memory=True)
    
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
