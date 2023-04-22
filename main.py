from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Process

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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


def train(seed, args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param seed : seed for random number generator
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    seed = seed
    training_acc = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        training_acc += pred.eq(target.view_as(pred)).sum().item()
        training_loss += F.nll_loss(output, target, reduction='sum').item()
        """The training loss and accuracy should be record by a file (e.g. TrainingRecord_seed.txt file) after each update."""
        with open(f"TrainingRecord_{seed}.txt", "a") as f:
            f.write("epoch: {}, batch: {}, loss: {}, accuracy: {} \n".format(epoch, batch_idx, training_loss, training_acc))
        """The training loss and accuracy should be print after each update."""
        print("epoch: {}, batch: {}, loss: {}, accuracy: {} \n".format(epoch, batch_idx, training_loss, training_acc))
    training_acc = 100. * training_acc / len(train_loader.dataset)
    training_loss = training_loss / len(train_loader.dataset)
    return training_acc, training_loss


def test(seed, model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param seed : seed for random number generator
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    seed = seed
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            """The testing loss, accuracy should be record by a file which separate by seed (e.g.TestingRecord_seed.txt file) after each update."""
            with open(f"TestingRecord_{seed}.txt", "a") as f:
                f.write("loss: {}, accuracy: {} \n".format(test_loss, correct))
    testing_acc = correct / len(test_loader.dataset)
    testing_loss = test_loss / len(test_loader.dataset)
    return testing_acc, testing_loss



def plot(epoches, performance):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    import matplotlib.pyplot as plt
    plt.plot(epoches, performance)


def run(seed, config):
    import matplotlib.pyplot as plt
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(seed, config, model, device, train_loader, optimizer, epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss = test(seed, model, device, test_loader)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        scheduler.step()
        epoches.append(epoch)

    """plotting each seed training performance with the records"""
    plt.plot(epoches, training_accuracies)
    plt.xlabel('Epoches')
    plt.ylabel('Training accuracy')
    plt.title(f"{seed}_training accuracy")
    plt.show()

    plt.plot(epoches, training_loss)
    plt.xlabel('Epoches')
    plt.ylabel('Training loss')
    plt.title(f"{seed}_training loss")
    plt.show()

    """plotting testing performance with the records"""
    plt.plot(epoches, testing_accuracies)
    plt.xlabel('Epoches')
    plt.ylabel('Testing accuracy')
    plt.title(f"{seed}_testing accuracy")
    plt.show()

    plt.plot(epoches, testing_loss)
    plt.xlabel('Epoches')
    plt.ylabel('Testing loss')
    plt.title(f"{seed}_testing loss")
    plt.show()

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")



def plot_mean():
    """
    Read each seed's recorded results.
    Plot the mean results after three runs.
    :return:
    """
    import matplotlib.pyplot as plt
    
    """read the recorded results"""
    epoches=[]
    training_accuracies_123 = []
    training_loss_123 = []
    testing_accuracy_123 = []
    testing_loss_123 = []
    training_accuracies_321 = []
    training_loss_321 = []
    testing_accuracy_321 = []
    testing_loss_321 = []
    training_accuracies_666 = []
    training_loss_666 = []
    testing_accuracy_666 = []
    testing_loss_666 = []
    training_accuracies_123_mean = []
    training_loss_123_mean = []
    testing_accuracy_123_mean = []
    testing_loss_123_mean = []
    training_accuracies_321_mean = []
    training_loss_321_mean = []
    testing_accuracy_321_mean = []
    testing_loss_321_mean = []
    training_accuracies_666_mean = []
    training_loss_666_mean = []
    testing_accuracy_666_mean = []
    testing_loss_666_mean = []
    training_accuracies_total_mean = []
    training_loss_total_mean = []
    testing_accuracy_total_mean = []
    testing_loss_total_mean = []
    with open("TrainingRecord_123.txt", "r") as f:
        for line in f:
            """Record the epoches"""
            if line.split(",")[0].split(" ")[1] not in epoches:
                epoches.append(line.split(",")[0].split(" ")[1])
            training_accuracies_123.append(line.split(", ")[3].split(" ")[1])
            training_loss_123.append(line.split(", ")[2].split(" ")[1])
    with open("TestingRecord_123.txt", "r") as f:
        for line in f:
            testing_loss_123.append(line.split(",")[0].split(" ")[1])
            testing_accuracy_123.append(line.split(", ")[1].split(" ")[1])
    with open("TrainingRecord_321.txt", "r") as f:
        for line in f:
            training_loss_321.append(line.split(", ")[2].split(" ")[1])
            training_accuracies_321.append(line.split(", ")[3].split(" ")[1])
    with open("TestingRecord_321.txt", "r") as f:
        for line in f:
            testing_loss_321.append(line.split(",")[0].split(" ")[1])
            testing_accuracy_321.append(line.split(", ")[1].split(" ")[1])
    with open("TrainingRecord_666.txt", "r") as f:
        for line in f:
            training_loss_666.append(line.split(", ")[2].split(" ")[1])
            training_accuracies_666.append(line.split(", ")[3].split(" ")[1])
    with open("TestingRecord_666.txt", "r") as f:
        for line in f:
            testing_loss_666.append(line.split(",")[0].split(" ")[1])
            testing_accuracy_666.append(line.split(", ")[1].split(" ")[1])

    """ Calculate the mean results of training_accuracies_seed, training_loss_seed, testing_accuracy_seed, testing_loss_seed"""
    for i in range(0, len(training_accuracies_123), 938):
        training_accuracies_123_mean.append(sum([float(training_accuracies_123[j]) for j in range(i, i+938)])/938)
        training_loss_123_mean.append(sum([float(training_loss_123[j]) for j in range(i, i+938)])/938)
    for i in range(0, len(testing_accuracy_123), 10):
        testing_accuracy_123_mean.append(sum([float(testing_accuracy_123[j]) for j in range(i, i+10)])/10)
        testing_loss_123_mean.append(sum([float(testing_loss_123[j]) for j in range(i, i+10)])/10)
    for i in range(0, len(training_accuracies_321), 938):
        training_accuracies_321_mean.append(sum([float(training_accuracies_321[j]) for j in range(i, i+938)])/938)
        training_loss_321_mean.append(sum([float(training_loss_321[j]) for j in range(i, i+938)])/938)
    for i in range(0, len(testing_accuracy_321), 10):
        testing_accuracy_321_mean.append(sum([float(testing_accuracy_321[j]) for j in range(i, i+10)])/10)
        testing_loss_321_mean.append(sum([float(testing_loss_321[j]) for j in range(i, i+10)])/10)
    for i in range(0, len(training_accuracies_666), 938):
        training_accuracies_666_mean.append(sum([float(training_accuracies_666[j]) for j in range(i, i+938)])/938)
        training_loss_666_mean.append(sum([float(training_loss_666[j]) for j in range(i, i+938)])/938)
    for i in range(0, len(testing_accuracy_666), 10):
        testing_accuracy_666_mean.append(sum([float(testing_accuracy_666[j]) for j in range(i, i+10)])/10)
        testing_loss_666_mean.append(sum([float(testing_loss_666[j]) for j in range(i, i+10)])/10)
    for i in range(0, len(training_accuracies_123_mean)):
        training_accuracies_total_mean.append((float(training_accuracies_123_mean[i])+float(training_accuracies_321_mean[i])+float(training_accuracies_666_mean[i]))/3)
        training_loss_total_mean.append((float(training_loss_123_mean[i])+float(training_loss_321_mean[i])+float(training_loss_666_mean[i]))/3)
    for i in range(0, len(testing_accuracy_123_mean)):
        testing_accuracy_total_mean.append((float(testing_accuracy_123_mean[i])+float(testing_accuracy_321_mean[i])+float(testing_accuracy_666_mean[i]))/3)
        testing_loss_total_mean.append((float(testing_loss_123_mean[i])+float(testing_loss_321_mean[i])+float(testing_loss_666_mean[i]))/3)
    """plot the results"""
    plt.plot(epoches, training_accuracies_123_mean, color='red', label='training accuracy_123')
    plt.plot(epoches, training_accuracies_321_mean, color='blue', label='training accuracy_321')
    plt.plot(epoches, training_accuracies_666_mean, color='green', label='training accuracy_666')
    plt.plot(epoches, training_accuracies_total_mean, color='black', label='training accuracy_mean')
    plt.xlabel('Epoches')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.show()

    plt.plot(epoches, training_loss_123_mean, color='red', label='training loss_123')
    plt.plot(epoches, training_loss_321_mean, color='blue', label='training loss_321')
    plt.plot(epoches, training_loss_666_mean, color='green', label='training loss_666')
    plt.plot(epoches, training_loss_total_mean, color='black', label='training loss_mean')
    plt.xlabel('Epoches')
    plt.ylabel('Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

    plt.plot(epoches, testing_accuracy_123_mean, color='red', label='testing accuracy_123')
    plt.plot(epoches, testing_accuracy_321_mean, color='blue', label='testing accuracy_321')
    plt.plot(epoches, testing_accuracy_666_mean, color='green', label='testing accuracy_666')
    plt.plot(epoches, testing_accuracy_total_mean, color='black', label='testing accuracy_mean')
    plt.xlabel('Epoches')
    plt.ylabel('Testing accuracy')
    plt.title('Testing accuracy')
    plt.legend()
    plt.show()

    plt.plot(epoches, testing_loss_123_mean, color='red', label='testing loss_123')
    plt.plot(epoches, testing_loss_321_mean, color='blue', label='testing loss_321')
    plt.plot(epoches, testing_loss_666_mean, color='green', label='testing loss_666')
    plt.plot(epoches, testing_loss_total_mean, color='black', label='testing loss_mean')
    plt.xlabel('Epoches')
    plt.ylabel('Testing loss')
    plt.title('Testing loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    arg = read_args()
    config = load_config(arg)
    """ define the processes to run the model"""
    processes = []
    """ run the model with different seeds in different processes"""
    for seed in config.seeds:
        p = Process(target=run, args=(seed, config))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    plot_mean()