#This code is adopted from tutoiral
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#The Residual Convolutional Network we chose to implement is called ResNet.
#The detailed explaination of ResNet has written in the report.
#Modified By Group 5 for experiments by tuning hyperparameters of CNN.
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from time import time
from torch.autograd import Variable


# ResNet Residual Block for layer size 18 and 34
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, act='ReLu'):
        super(ResidualBlock, self).__init__()         

        # Residual Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Residual Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Define activation function
        if act == 'ReLu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
           self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                         nn.BatchNorm2d(out_channels))

    def forward(self, x):
        
        # First Pass x through the first Conv Net.
        # Apply Batch Normalization on the result.
        out = self.act(self.bn1(self.conv1(x)))

        # Pass x through the second Conv Net.
        # Apply Batch Normalization on the result.
        out = self.bn2(self.conv2(out))

        # Pass x through the shortcut.
        # z(x) = x + f(x)
        out += self.shortcut(x)

        # Pass x through the activ.
        out = self.act(out)
        return out


# ResNet Residual Block for layer size 50,101 and 152
class ResidualBlock50_101_152(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, base_width=64, downsample=None, act='ReLu'):
        super(ResidualBlock50_101_152, self).__init__()         

        # Residual Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Residual Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        # Define activation function
        if act == 'ReLu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        
        # First Pass x through the first Conv Net.
        # Apply Batch Normalization on the result.
        out = self.act(self.bn1(self.conv1(x)))

        # Pass x through the second Conv Net.
        # Apply Batch Normalization on the result.
        out = self.act(self.bn2(self.conv2(out)))

        # Pass x through the third Conv Net.
        # Apply Batch Normalization on the result.
        out = self.bn3(self.conv3(out))

        # Pass x through the shortcut.
        # z(x) = x + f(x)
        out += self.shortcut(x)

        # Pass x through the activ.
        out = self.act(out)
        return out


#Create CNN Classifier:
class ResNet(nn.Module):
    activation = 'ReLu'
    def __init__(self, block, layers, num_classes=10):      
        # Initialize ResNet Class
        super(ResNet, self).__init__()
        # Initialize input conv
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, bias=False)

        # Batch Normalization by BatchNorm2d.
        # computes the mean and standard deviation per channel/Dimension,
        # since Convolution has 64 out_channels, then the batch will take an
        # input dimension as 64
        self.bn1 = nn.BatchNorm2d(64)

        #Define Activation Function Relu
        if self.activation == 'ReLu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()

        #Define Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Create 4 ResNet layers, by the given size of the layers:
        # For 18 layers: [2(64 + 64), 2(128 + 128), 2(256 + 256), 2(512 + 512)]
        # For 34 layers: [3(64 + 64), 4(128 + 128), 6(256 + 256), 3(512 + 512)]
        # For 50 layers: [3(64 + 64 + 256), 4(128 + 128 + 512), 6(256 + 256 +
        # 1024), 3(512 + 512 + 2048)]
        # For 101 layers: [3(64 + 64 + 256), 4(128 + 128 + 512), 23(256 + 256 +
        # 1024), 3(512 + 512 + 2048)]
        # For 152 layers: [3(64 + 64 + 256), 8(128 + 128 + 512), 36(256 + 256 +
        # 1024), 3(512 + 512 + 2048)]
        # First Layer with base 64 out_channels
        self.layer1 = self._create_res_layer(block, 64, layers[0])
        # Second Layer with base 128 out_channels
        self.layer2 = self._create_res_layer(block, 128, layers[1], stride=2)
        # Thrid Layer with base 256 out_channels
        self.layer3 = self._create_res_layer(block, 256, layers[2], stride=2)
        # Forth Layer with base 512 out_channels
        self.layer4 = self._create_res_layer(block, 512, layers[3], stride=2)

        #Define Avg pooling layer
        self.avgpool = nn.AvgPool2d(1, stride=1)

        # Then, the fully connected Layer:
        self.fc = nn.Linear(512 * block.expansion, num_classes)
     
    # A layer is just two residual blocks for ResNet
    def _create_res_layer(self, block, out_channels, res_layers, stride=1):
        # Create layers and append to array
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        # The input channel size is the size of the input mulitply by the
        # expansion.
        # For instance, the 50 resNet will require an output channels of 4
        # times the input channels size.
        self.in_channels = out_channels * block.expansion
        # Create layers by the given architecture
        for i in range(1, res_layers):
            layers.append(block(self.in_channels, out_channels))

        #Sequential is a container of Modules that can be stacked together and
        #run at the same time.
        r"""
        For example: we have two Residual Blocks in Sequences, so that we can stack layers together.
        (layer3): Sequential(
            (0): ResidualBlock(
              (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (shortcut): Sequential(
                (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBlock(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (shortcut): Sequential()
            )
          )
        """
        return nn.Sequential(*layers)

    # Feed data input x into the network
    def forward(self, x):
        # First Pass x through the first Conv Net.
        # Apply Batch Normalization on the result.
        x = self.conv1(x)
        x = self.bn1(x)   
        #Apply ReLu as activation Function:
        x = self.act(x)
        # Current output size: 112x112

        # First layers of ResNet
        x = self.maxpool(x)  
        x = self.layer1(x) 
        # Current output size: 56x56

        # Second layers of ResNet
        x = self.layer2(x)
        # Current output size: 28x28

        # Third layers of ResNet
        x = self.layer3(x)
        # Current output size: 14x14

        # Forth layers of ResNet
        x = self.layer4(x)   # 7x7
        # Current output size: 7x7

        # Finally, applying average pooling
        x = self.avgpool(x)  
        # Current output size: 1x1

        # Apply the final fully connect layer fc1
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def simple_output(testloader, clf, images, labels, classes):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        outputs = clf(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

    print(' '.join('label: %5s, predict: %5s.' % (classes[labels[j]], classes[predicted[j].item()]) for j in range(len(labels))))


def plot_tvt_acc(layer, number_of_epoch, train_acc, val_acc, test_acc, lr, size_of_batch_train, optimizer, activation):
    plt.figure()
    plt.plot([i + 1 for i in range(number_of_epoch)], train_acc, marker='*', label='train accuracy', color = 'blue')
    plt.plot([i + 1 for i in range(number_of_epoch)], val_acc, marker='o',label='validation accuracy', color = 'black')
    plt.plot([i + 1 for i in range(number_of_epoch)], test_acc, marker='s',label='test accuracy', color = 'red')
    plt.grid()
    plt.legend(loc = 'lower right')
    plt.yticks(np.arange(0.3,1.1,0.1))
    plt.title("ResNet" + str(layer) + ', Act:' + activation + ', Opt:' + optimizer + ', LR: ' + str(lr) + ', Batch Size: ' + str(size_of_batch_train))
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("ResNet-" + str(layer) + activation  + optimizer + "Acc.png")


def plot_all_acc(number_of_epoch, all_test_acc, lr, size_of_batch_train, optimizer = 'Adam', activation = 'ReLu', job = []):
    
    if len(job) != 0:       
        plt.figure()
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[0], marker='*', ms=5, label= job[0] + ' Test Accuracy', color = 'r')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[1], marker='o', ms=5, label= job[1] + ' Test Accuracy', color = 'k')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[2], marker='s', ms=5, label= job[2] + ' Test Accuracy', color = 'g')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[3], marker='+', ms=5, label= job[3] + ' Test Accuracy', color = 'y')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[4], marker='h', ms=5, label= job[4] + ' Test Accuracy', color = 'b')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[5], marker='x', ms=5, label= job[5] + ' Test Accuracy', color = 'm')
        plt.grid()
        plt.legend(loc = 'lower right')
        plt.yticks(np.arange(0.3,0.85,0.05))
        plt.title('ResNet 18 Optimal Accuracy, Batch Size: ' + str(size_of_batch_train))
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy')
        plt.savefig("ResNet-Acc.png")
    else:
        plt.figure()
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[0], marker='*', label= optimizer + '-' + activation + ' Test Accuracy', color = 'blue')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[1], marker='o', label= optimizer + '-' + activation + ' Test Accuracy', color = 'black')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_test_acc[2], marker='s', label= optimizer + '-' + activation + ' Test Accuracy', color = 'red')
        plt.grid()
        plt.legend(loc = 'lower right')
        plt.yticks(np.arange(0.3,1.0,0.1))
        plt.title("ResNet 18 Acc, " + ', Act:' + activation + ', Opt:' + optimizer + ', LR: ' + str(lr) + ', Batch Size: ' + str(size_of_batch_train))
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy')
        plt.savefig("ResNet-Acc-"+ activation + optimizer + ".png")

def plot_mean_cost(number_of_epoch, all_mean_cost, lr, size_of_batch_train, optimizer = 'Adam', activation = 'ReLu', job = []):

    if len(job) != 0:   
        plt.figure()
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[0], marker='*', ms=5, label= job[0] + ' Mean Cost', color = 'r')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[1], marker='o', ms=5, label= job[1] + ' Mean Cost', color = 'k')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[2], marker='s', ms=5, label= job[2] + ' Mean Cost', color = 'g')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[3], marker='+', ms=5, label= job[3] + ' Mean Cost', color = 'y')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[4], marker='h', ms=5, label= job[4] + ' Mean Cost', color = 'b')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[5], marker='x', ms=5, label= job[5] + ' Mean Cost', color = 'm')
        plt.grid()
        plt.legend(loc = 'upper right')
        #plt.yticks(np.arange(0,2.4,0.1))
        plt.title("ResNet 18 Optimal Mean Cost, Batch Size: " + str(size_of_batch_train))
        plt.xlabel('Training Epoch')
        plt.ylabel('Mean Cost')
        plt.savefig("ResNet-Mean.png")
    else:
        plt.figure()
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[0], marker='*', label=optimizer + '-' + activation + ' Mean Cost', color = 'blue')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[1], marker='o', label=optimizer + '-' + activation + ' Mean Cost', color = 'black')
        plt.plot([i + 1 for i in range(number_of_epoch)], all_mean_cost[2], marker='s', label=optimizer + '-' + activation + ' Mean Cost', color = 'red')
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.title("ResNet 18 Mean," + ', Act:' + activation + ', Opt:' + optimizer + ', LR: ' + str(lr) + ', Batch Size: ' + str(size_of_batch_train))
        plt.xlabel('Training Epoch')
        plt.ylabel('Mean Cost')
        plt.savefig("ResNet-Mean-"+ activation + optimizer + ".png")

def plot_from_log():
    f = open("p3.log", "r")
    lines = f.read().splitlines() 
    all_mean_cost = []
    all_test_acc = []
    mean_cost = []
    test_acc = []
    train_acc = []
    val_acc = []
    layers = [152, 50, 18]
    k = 0
    for raw_line in lines:
        line = str(raw_line)       
        if 'mean' in line:
            cost_str = line.split(':')[2]
            mean_cost.append(float(cost_str))
        if 'Test Acc' in line:
            cost_str = line.split(':')[1].split('%')[0]
            test_acc_float = float(cost_str) / 100.
            test_acc.append(test_acc_float)
        if 'Validation Acc' in line:
            cost_str = line.split(':')[1].split('%')[0]
            val_acc_float = float(cost_str) / 100.
            val_acc.append(val_acc_float)
        if 'Traing Acc' in line:
            cost_str = line.split(':')[1].split('%')[0]
            train_acc_float = float(cost_str) / 100.
            train_acc.append(train_acc_float)

        if 'Finished,' in line:      
            plot_tvt_acc(layers[k], 100, train_acc, val_acc, test_acc, 0.001, 128, 'Adam', 'ReLu')
            all_mean_cost.insert(k, mean_cost)
            all_test_acc.insert(k, test_acc)
            k += 1
            mean_cost = []
            test_acc = []
            train_acc = []
            val_acc = []     

    plot_all_acc(100, all_test_acc, 0.001, 128, 'Adam', 'ReLu')
    plot_mean_cost(100, all_mean_cost, 0.001, 128, 'Adam', 'ReLu')


    f.close()

def plot_from_log2():
    f = open("log_1.txt", "r")
    lines = f.read().splitlines() 
    all_mean_cost = []
    all_test_acc = []
    mean_cost = []
    test_acc = []
    train_acc = []
    val_acc = []
    k = 0
    max_test = 0.0
    for raw_line in lines:
        line = str(raw_line)       
        if 'mean' in line:
            cost_str = line.split(':')[2]
            mean_cost.append(float(cost_str))
        if 'Test Acc' in line:
            cost_str = line.split(':')[1].split('%')[0]
            test_acc_float = float(cost_str) / 100.
            test_acc.append(test_acc_float)
            if test_acc_float > max_test:
                max_test = test_acc_float
        if 'Validation Acc' in line:
            cost_str = line.split(':')[1].split('%')[0]
            val_acc_float = float(cost_str) / 100.
            val_acc.append(val_acc_float)
        if 'Traing Acc' in line:
            cost_str = line.split(':')[1].split('%')[0]
            train_acc_float = float(cost_str) / 100.
            train_acc.append(train_acc_float)

        if 'Finished' in line:
            if k == 0:
                plot_tvt_acc(18, 100, train_acc, val_acc, test_acc, 0.001, 128, 'SGD', 'ReLu')
            if k == 1:
                plot_tvt_acc(18, 100, train_acc, val_acc, test_acc, 0.001, 128, 'Adam', 'ReLu')
            if k == 2:
                plot_tvt_acc(18, 100, train_acc, val_acc, test_acc, 0.001, 128, 'Adagrad', 'ReLu')
            if k == 3:
                plot_tvt_acc(18, 100, train_acc, val_acc, test_acc, 0.001, 128, 'SGD', 'Sigmoid')
            if k == 4:
                plot_tvt_acc(18, 100, train_acc, val_acc, test_acc, 0.001, 128, 'Adam', 'Sigmoid')
            if k == 5:
                plot_tvt_acc(18, 100, train_acc, val_acc, test_acc, 0.001, 128, 'Adagrad', 'Sigmoid')

            all_mean_cost.insert(k, mean_cost)
            all_test_acc.insert(k, test_acc)
            k += 1
            mean_cost = []
            test_acc = []
            train_acc = []
            val_acc = []     
    print(max_test)
    plot_all_acc(100, all_test_acc, 0.001, 128, job = ['SGD-ReLu', 'Adam-ReLu', 'Adagrad-ReLu', 'SGD-Sigmoid', 'Adam-Sigmoid', 'Adagrad-Sigmoid'])
    plot_mean_cost(100, all_mean_cost, 0.001, 128,  job = ['SGD-ReLu', 'Adam-ReLu', 'Adagrad-ReLu', 'SGD-Sigmoid', 'Adam-Sigmoid', 'Adagrad-Sigmoid'])


    f.close()


def predict(clf, device, testloader):
    # set mode to evluation
    clf.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # predict test data, we do need to calculate the data, so we use no_grad()
    # method.
    with torch.no_grad():
        # load test dataset
        
        for batch_idx, (images, labels) in enumerate(testloader):
            # take data and labels
            images, labels = images.to(device), labels.to(device)
            # predict/test the data
            outputs = clf(images)
            # the prediction class, ignore the first output
            _, predicted = torch.max(outputs, 1)
            #squeezing, to remove single-dimensional entries from the shape of
            #an array.
            c = (predicted == labels).squeeze()

            # For each batch, put predictions into arraies for further analysis
            for i in range(len(labels)):
                # Get the current label
                label = labels[i]
                # add one to the correct prediction
                class_correct[label] += c[i].item()
                # add one to the total number
                class_total[label] += 1
    ##Creating Labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct = 0
    total = 0
    #Find Total Accuracy
    for i in range(10):
        #print('total number of test data of %5s is: %d, correct prediction is:
        #%d, and accuracy is : %2d %%' % (classes[i], class_total[i],
        #class_correct[i], 100 * class_correct[i] / class_total[i]))
        correct += class_correct[i]
        total += class_total[i]

    return correct, total, class_correct, class_total
                            
if __name__ == "__main__":
    plot_from_log()
    plot_from_log2()
    f = open("log_2.txt", "w")
    number_of_epoch = 100
    size_of_batch_train = 128
    lr = 0.001
    #Creating the rules for transforming train dataset into Tensor.
    #Loading Training dataset, applying random horizontal flips to some images
    #####OPTIONAL: transforms.RandomCrop(32, padding=4)
    cifar_train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])    
    train_and_validation_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_train_transform)

    #Creating the rules for transforming test dataset into Tensor.
    #Loading Test dataset
    cifar_test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=size_of_batch_train, shuffle=False, num_workers=2)


    #Train and Validation split
    train_percentage = 0.95
    np.random.seed(42)
    train_val_split = np.random.permutation(len(train_and_validation_set))
    #Split the data into training and validation set
    train = torch.utils.data.Subset(train_and_validation_set,train_val_split[:int(train_percentage * len(train_and_validation_set))])
    validation = torch.utils.data.Subset(train_and_validation_set,train_val_split[int(train_percentage * len(train_and_validation_set)):])
    #Create the data loader
    trainloader = torch.utils.data.DataLoader(train, batch_size=size_of_batch_train, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(validation, batch_size=size_of_batch_train, shuffle=True, num_workers=2)

    #Enable GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()
    #clfs = []
    #layers = [18, 18]
    #create the ResNet-152
    #clf152_Relu = ResNet(ResidualBlock50_101_152, [3,8,36,3])
    #clf152_Sigm = ResNet(ResidualBlock50_101_152, [3,8,36,3])
    #clf152_Sigm.activation = 'Sigmoid'
    #clfs.append(clf152_Relu)
    #clfs.append(clf152_Sigm)
    #create the ResNet-50
    #clf50_Relu = ResNet(ResidualBlock50_101_152, [3,4,6,3])
    #clf50_Sigm = ResNet(ResidualBlock50_101_152, [3,4,6,3])
    #clf50_Sigm.activation = 'Sigmoid'
    #clfs.append(clf50_Relu)
    #clfs.append(clf50_Sigm)
    #create the ResNet-18
    #clf18_Relu = ResNet(ResidualBlock, [2,2,2,2])
    #clf18_Sigm = ResNet(ResidualBlock, [2,2,2,2])
    #Assign different Activition Function
    #clf18_Sigm.activation = 'Sigmoid'
    #clfs.append(clf18_Relu)
    #clfs.append(clf18_Sigm)

    #Define Loss function as Entropy Loss
    #criterion = nn.MSELoss(size_average=False)
    criterion = nn.CrossEntropyLoss()
    all_mean_cost = []
    all_test_acc = []
    job = []
    #op_names = ['SGD', 'Adam', 'Adagrad', 'RMSprop']
    op_names = ['Adagrad']
    
    for k in range(0, 2):        
        for op in range(len(op_names)):
            if k == 0:
                clf = ResNet(ResidualBlock, [2,2,2,2])
            else:
                clf = ResNet(ResidualBlock, [2,2,2,2])
                clf.activation = 'Sigmoid'

            #Sent the network to GPU
            clf.to(device)

            #Implements Adam and Stochastic gradient descent with momentum as optimizer
            #if op == 0:
            #    optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=0.99, weight_decay=5e-4)
            #if op == 1:
            #    optimizer = optim.Adam(clf.parameters(), lr=lr, betas=(0.5,0.999))
            #if op == 2:
            #    optimizer = optim.Adagrad(clf.parameters(), lr=lr, lr_decay=0, weight_decay=5e-4, initial_accumulator_value=0, eps=1e-10)
            #if op == 3:
            #    optimizer = optim.RMSprop(clf.parameters(), lr=lr)

            optimizer = optim.Adagrad(clf.parameters(), lr=lr, lr_decay=0, weight_decay=5e-4, initial_accumulator_value=0, eps=1e-10)
            train_acc = []
            val_acc = []
            test_acc = []
            mean_cost = []
            current_job = op_names[op] + '-' + clf.activation
            job.append(current_job)
            print(job)
            f.write(current_job + '\n')
            f.flush()
            #Train the network
            start = time()
            train_start = start
            for epoch in range(number_of_epoch):
                losses = []
                
                # train
                clf.train()
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    # pass the input data and target(label) to the GPU/CPU
                    inputs, targets = inputs.to(device), targets.to(device)
                    # Set the gradient to zero, otherwise the graident will be
                    # added to
                    # # the optimizer each time.
                    optimizer.zero_grad()

                    # A Variable wraps a Tensor.
                    # # It supports nearly all the API’s defined by a Tensor.
                    # # Variable also provides a backward method to perform
                    # backpropagation
                    inputs, targets = Variable(inputs), Variable(targets)

                    # Train the inputs
                    outputs = clf(inputs)

                    # Calculate the loss
                    loss = criterion(outputs, targets)
                    # and Perform backpropagation
                    loss.backward()
                    # and closure of the single optimization step
                    optimizer.step()
                    # append the loss to the array for further analysis
                    losses.append(loss.item())

                
                mean_cost_epoch = np.mean(losses)
                print('epoch:' + str(epoch) + ', mean:' + str(mean_cost_epoch))
                f.write('epoch:' + str(epoch) + ', mean:' + str(mean_cost_epoch) + '\n')
                f.flush()
                mean_cost.append(mean_cost_epoch)

                #Accuracy on the training set
                correct, total, class_correct, class_total = predict(clf, device, trainloader)
                print('Traing Acc : %.3f %%' % (100. * correct / total))
                f.write('Traing Acc : %.3f %%\n' % (100. * correct / total))
                f.flush()
                train_acc.append(1. * correct / total)

                #Accuracy on the validation set
                correct, total, class_correct, class_total = predict(clf, device, valloader)
                print('Validation Acc : %.3f %%' % (100. * correct / total))
                f.write('Validation Acc : %.3f %%\n' % (100. * correct / total))
                f.flush()
                val_acc.append(1. * correct / total)

                #Accuracy on the test set
                correct, total, class_correct, class_total = predict(clf, device, testloader)
                print('Test Acc : %.3f %%' % (100. * correct / total))
                f.write('Test Acc : %.3f %%\n' % (100. * correct / total))
                f.flush()
                test_acc.append(1. * correct / total)
            
            print('Finished, Total Training Time:', ((time() - train_start)), end='\n')
            f.write('Finished, Total Training Time:' + str(time() - train_start) + '\n')
            f.flush()
            all_mean_cost.insert(k, mean_cost)
            all_test_acc.insert(k, test_acc)
            #Save the result
            #path = './' + str(18) + op_names[op] + clf.activation +  '_cifar_net.pth'
            #torch.save(clf.state_dict(), path)
            #Plot the train, validation and test set accuracy
            plot_tvt_acc(18, number_of_epoch, train_acc, val_acc, test_acc, lr, size_of_batch_train, op_names[op], clf.activation)

          
    f.close()
    #plot_all_acc(number_of_epoch, all_test_acc, lr, size_of_batch_train, job = job)
    #plot_mean_cost(number_of_epoch, all_mean_cost, lr, size_of_batch_train, job = job)
    # load back trained network
    #clf = ResNet(ResidualBlock50_101_152, [3,4,6,3])
    #clf = Resnet(residualblock, [3,4,6,3])
    #clf.load_state_dict(torch.load('./cifar_net.pth'))         
    #simple_output(testloader, clf, images, labels, classes)