# CSC1004-Python-Image Classification

<h3> To-Do </h3>
<ul>
  <li><strike>Training Function (10%)</strike></li>
  <li><strike>Testing Function (10%)</strike></li>
  <li><strike>Plotting Function (20%)</strike></li>
  <li><strike>Random Seed (20%)</strike></li>
  <li><strike>Python Multi-Processing (30%)</strike></li>
</ul>

### Configuration:

```
batch_size: 64  #  input batch size for training
test_batch_size: 1000  # input batch size for testing
epochs: 15  # number of epochs to train
lr: 0.01  # learning rate
gamma: 0.7  # learning rate step gamma
no_cuda: True  # disables CUDA training
no_mps: True   # disables macOS GPU training
dry_run: False # quickly check a single pass
seeds: [123, 321, 666]  # random seeds for the three runs are 123, 321, 666
log_interval: 10  # how many batches to wait before logging training status
save_model: True  # For Saving the current Model
```

This is the configuration of the Python-Image Classification Project. In order to implement the Random Seed running, I change the minist.yaml provided in the course website, I used list to hold the three seeds 123, 321, 666. In the following function, the seeds list will be traversed to use each seed to run the training and testing function.

### Training Function:

```
def train(seed, args, model, device, train_loader, optimizer, epoch):
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
        with open(f"TrainingRecord_{seed}.txt", "a") as f:
            f.write("epoch: {}, batch: {}, loss: {}, accuracy: {} \n".format(epoch, batch_idx, training_loss, training_acc))
        print("epoch: {}, batch: {}, loss: {}, accuracy: {} \n".format(epoch, batch_idx, training_loss, training_acc))
    training_acc = 100. * training_acc / len(train_loader.dataset)
    training_loss = training_loss / len(train_loader.dataset)
    return training_acc, training_loss
```

This function trains the model for one epoch using the training data. It takes as input the seed number, input arguments, neural network model, device, training data loader, optimizer, and current epoch. It performs the forward pass, computes the loss, and updates the parameters using backpropagation. It also records the training accuracy and loss after each batch and saves them in a file with the seed number appended to the filename.

### Testing function:

```
def test(seed, model, device, test_loader):
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
            with open(f"TestingRecord_{seed}.txt", "a") as f:
                f.write("loss: {}, accuracy: {} \n".format(test_loss, correct))
    testing_acc = correct / len(test_loader.dataset)
    testing_loss = test_loss / len(test_loader.dataset)
    return testing_acc, testing_loss
```

This function evaluates the model using the testing data. It takes as input the seed number, neural network model, device, and testing data loader. It performs the forward pass and computes the testing accuracy and loss. It also saves the testing accuracy and loss in a file with the seed number appended to the filename.

### Plotting function:

```
def plot(epoches, performance):
    import matplotlib.pyplot as plt
    plt.plot(epoches, performance)
```

This function plots the performance of the model over the recorded epochs. It takes as input the recorded epoch numbers and performance.

```
import matplotlib.pyplot as plt
plt.plot(epoches, training_accuracies)
plt.xlabel('Epoches')
plt.ylabel('Training accuracy')
plt.title(f"{seed}_training accuracy")
plt.show()
```

In the Running function, the line charts are based on the generated training accuracy, training loss, testing loss, and testing accuracy (i.e., generate four plots) for each seed's run. It set the x-label 'Epoches', y-label 'Training or Testing accuracy or loss', and title 'seed_Training or Testing accuracy or loss'.

```
def plot_mean():
    import matplotlib.pyplot as plt
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
```

In the Plot_mean function, I first create several lists to record the read results. Since the recorded style of training and testing outcome is below:

#### Training sample.txt

```
epoch: 1, batch: 0, loss: 31.12604331970215, accuracy: 56 
epoch: 1, batch: 1, loss: 55.80928611755371, accuracy: 115 
epoch: 1, batch: 2, loss: 81.32924461364746, accuracy: 174 
epoch: 1, batch: 3, loss: 114.55346870422363, accuracy: 230 
epoch: 1, batch: 4, loss: 145.93931770324707, accuracy: 279 
epoch: 1, batch: 5, loss: 171.86688995361328, accuracy: 335 
epoch: 1, batch: 6, loss: 196.28451919555664, accuracy: 391 
epoch: 1, batch: 7, loss: 225.29649353027344, accuracy: 447 
epoch: 1, batch: 8, loss: 256.6288833618164, accuracy: 502 
epoch: 1, batch: 9, loss: 282.1397132873535, accuracy: 559 
epoch: 1, batch: 10, loss: 300.94307708740234, accuracy: 618 
epoch: 1, batch: 11, loss: 330.7225399017334, accuracy: 673 
epoch: 1, batch: 12, loss: 350.8218517303467, accuracy: 731 
epoch: 1, batch: 13, loss: 374.62659072875977, accuracy: 788 
epoch: 1, batch: 14, loss: 394.30347633361816, accuracy: 847 
epoch: 1, batch: 15, loss: 423.38061141967773, accuracy: 901 
```

#### Testing sample.txt

```
loss: 276.020751953125, accuracy: 922 
loss: 560.5607299804688, accuracy: 1835 
loss: 842.6048278808594, accuracy: 2754 
loss: 1134.4953308105469, accuracy: 3668 
loss: 1408.6819152832031, accuracy: 4592 
loss: 1673.8036499023438, accuracy: 5517 
loss: 1923.6719512939453, accuracy: 6444 
loss: 2183.928421020508, accuracy: 7377 
loss: 2449.8502655029297, accuracy: 8301 
loss: 2770.856887817383, accuracy: 9216 
loss: 201.42051696777344, accuracy: 945 
loss: 432.8534698486328, accuracy: 1878 
loss: 618.3109436035156, accuracy: 2823 
loss: 843.7655792236328, accuracy: 3756 
loss: 1052.6648406982422, accuracy: 4691 
loss: 1277.5500793457031, accuracy: 5619 
```

Therefore I use the **'split'** and **'index'** to read the recorded results and save them in the lists. Then I do some simple calculations to get the mean result of each seed in each epoch and test. After that, I calculate the mean result of the three seeds and plot each seed line chart and the mean of each seed to compare with each other. The black line chart is the mean one that satisfied the requirement. 

### Random seed and Mult-Processing

```
processes = []
for seed in config.seeds:
    p = Process(target=run, args=(seed, config))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
```

To implement the random seed and Mult-Processing requirement, in the '**main**' part I first create a processes list. The processes list is created to store references to the processes that will be created in the loop. The loop iterates over each seed in the config, seeds, and creates a new Process object using the Process constructor. The target argument specifies the function to be run in the new process, which is the run function. The newly created Process object is appended to the processes list, and then the start method is called on the Process object. This starts the execution of the run function in a new process. The join() method blocks the main program until the corresponding process has completed execution.  This ensures that the program does not terminate until all the processes have completed their tasks.

### Plotting sample

![5fe1634bf739d72df35c92dd79347c6](C:\Users\17793\AppData\Local\Temp\WeChat Files\5fe1634bf739d72df35c92dd79347c6.png)

![99cb4510b0d6486a1784293ea3c48cf](C:\Users\17793\AppData\Local\Temp\WeChat Files\99cb4510b0d6486a1784293ea3c48cf.png)

![82bf83e9c14b4ff8fab56e8ff70b825](C:\Users\17793\AppData\Local\Temp\WeChat Files\82bf83e9c14b4ff8fab56e8ff70b825.png)

![408f3bc14f59e9cc55c6ee8a7189950](C:\Users\17793\AppData\Local\Temp\WeChat Files\408f3bc14f59e9cc55c6ee8a7189950.png)

![6ce9a1979a94bbeab4a66609024078c](C:\Users\17793\AppData\Local\Temp\WeChat Files\6ce9a1979a94bbeab4a66609024078c.png)

![a6c677465f4d196dc7bd59e854d58a7](C:\Users\17793\AppData\Local\Temp\WeChat Files\a6c677465f4d196dc7bd59e854d58a7.png)

![24f1f6251f57ef204cd4b8a936436d8](C:\Users\17793\AppData\Local\Temp\WeChat Files\24f1f6251f57ef204cd4b8a936436d8.png)

![152e8617cdc4dbf354447b0a72b18f2](C:\Users\17793\AppData\Local\Temp\WeChat Files\152e8617cdc4dbf354447b0a72b18f2.png)