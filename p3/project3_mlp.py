import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class Linear_layer:
    def __init__(self, input_dimension, output_dimension, learning_rate):
        self.learning_rate = learning_rate
        # Random initialization of weight and bias
        # Xavier initialization: weight (mean = 0 and variance = 2/(input_dimension + output_dimension)
        self.weight = np.random.randn(input_dimension,output_dimension)*np.sqrt(1/input_dimension)
        #self.weight = np.random.normal(loc=0.0, scale = (2/(input_dimension+output_dimension))**0.5, size = (input_dimension,output_dimension))
        #self.weight = np.ones((input_dimension,output_dimension))
        self.bias = np.zeros(output_dimension)[None,:]
        
    def forward(self,input_of_Linear_layer):
        # output = input*W+b
        # input shape = batch_size * input_dimension
        # output shape = batch_size * out_dimension
        # weight shape = input_dimentsion * output_dimension
        # bias shape = 1 * output_dimension
        
        return np.dot(input_of_Linear_layer,self.weight) + self.bias

    def backward(self,input_of_Linear_layer,df_over_doutput):
        # using the chain rule: df/dinput = df/doutput * doutput/dinput
        # doutput/dinput = weight.T
        df_over_dinput = np.dot(df_over_doutput, self.weight.T)
        
        # df/dweight = doutput/dweight * df/doutput
        # doutput/dweight = input.T
        df_over_dweight = np.dot(input_of_Linear_layer.T, df_over_doutput)

        df_over_dbias = np.mean(df_over_doutput, axis=0)[None,:]
        

        # Updata weight and bias using GD
        self.weight = self.weight - self.learning_rate * df_over_dweight
        self.bias = self.bias - self.learning_rate * df_over_dbias
        
        return df_over_dinput

class ReLU:
    def forward(self, input_of_ReLU):
        return np.maximum(input_of_ReLU,0) 

    
    def backward(self, input_of_ReLU, df_over_doutput):
        # df_over_dinput = np.zeros(df_over_doutput.shape)

        
        # [rows, cols] = df_over_dinput.shape
        # for i in range(rows):
        #     for j in range(cols):
        #         if input_of_ReLU[i][j]<0:
        #             df_over_dinput[i][j] = 0.0
        #         else:
        #             df_over_dinput[i][j] = df_over_doutput[i][j]

        return (input_of_ReLU > 0) * df_over_doutput


class Leaky_ReLU:
    def __init__(self, initial_gamma, learning_rate):
        self.learning_rate = learning_rate
        self.gamma = initial_gamma

    def forward(self, input_of_Leaky_ReLU):
        return np.maximum(input_of_Leaky_ReLU,0) + self.gamma * np.minimum(input_of_Leaky_ReLU,0)


    def backward(self, input_of_Leaky_ReLU, df_over_doutput):
        df_over_dgamma = np.mean(np.sum(np.minimum(input_of_Leaky_ReLU, 0) * df_over_doutput, axis = 1))
        self.gamma = self.gamma - self.learning_rate * df_over_dgamma
        if self.gamma < 0 :
            self.gamma = 0
        df_over_dinput = (input_of_Leaky_ReLU > 0) * df_over_doutput + self.gamma * (input_of_Leaky_ReLU <= 0) * df_over_doutput
        return df_over_dinput



def softmax(input_of_softmax):
    input_exp = np.exp(input_of_softmax - np.max(input_of_softmax, 1)[:, None]) 
    return input_exp / np.sum(input_exp, axis=-1)[:, None]


def logsumexp(Z):
    Zmax = np.max(Z,axis=1)[:, None]
    return Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=1))[:, None] 


def mean_cost_crossentropy(input_of_softmax, reference_label):
    reference_matrix = np.zeros(input_of_softmax.shape, dtype = int)
    for i in range(len(reference_label)):
        reference_matrix[i][reference_label[i]] = 1
    return - np.mean(np.sum(input_of_softmax*reference_matrix, 1)[:, None]  - logsumexp(input_of_softmax))

def calculate_dmean_cost_crossentropy_over_dinputofSoftmax(input_of_softmax,reference_label):
    reference_matrix = np.zeros(input_of_softmax.shape, dtype = int)
    for i in range(len(reference_label)):
        reference_matrix[i][reference_label[i]] = 1
    return (-reference_matrix + softmax(input_of_softmax))/input_of_softmax.shape[0]




def forward(nn, input_of_nn):
    output_of_each_layer = []
    current_input = input_of_nn
    for i in range(len(nn)):
        output_of_each_layer.append(nn[i].forward(current_input))
        current_input = output_of_each_layer[-1] #nn[i].forward(current_input)
    return output_of_each_layer

# def forward2(nn, input_of_nn):
#     output_of_each_layer = []
#     current_input = input_of_nn
#     output_of_each_layer.append(nn[0].forward(current_input))
#     current_input = output_of_each_layer[-1] 
#     output_of_each_layer.append(ReLU().forward(current_input))
#     current_input = output_of_each_layer[-1] 


#     # for layer in nn:
#     #     output_of_each_layer.append(layer.forward(current_input))
#     #     current_input = output_of_each_layer[-1]
    
#     return output_of_each_layer[0]

def predict(nn,input_of_nn):
    result_matrix = forward(nn,input_of_nn)[-1]
    return (np.argmax(result_matrix,axis =1))


def train(nn,input_of_nn,reference_label):

    input_of_each_layer = [input_of_nn]+forward(nn,input_of_nn)
    input_of_softmax = input_of_each_layer[-1]
    
    mean_cost = mean_cost_crossentropy(input_of_softmax,reference_label)
    dmean_cost_over_dinput_of_softmax = calculate_dmean_cost_crossentropy_over_dinputofSoftmax(input_of_softmax,reference_label)
    dmean_cost_over_dcurrent_input = dmean_cost_over_dinput_of_softmax

    for i in range(len(nn)):
        layer_index = len(nn)-1-i

        layer = nn[layer_index]
        
        dmean_cost_over_dcurrent_input = layer.backward(input_of_each_layer[layer_index],dmean_cost_over_dcurrent_input) 
        
    return mean_cost







transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



train_and_validation_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

train_percentage = 0.95
np.random.seed(42)
train_val_split = np.random.permutation(len(train_and_validation_set))
trainset = torch.utils.data.Subset(train_and_validation_set,train_val_split[:int(train_percentage*len(train_and_validation_set))])
valset = torch.utils.data.Subset(train_and_validation_set,train_val_split[int(train_percentage*len(train_and_validation_set)):])

#print(len(trainset))

#trainset, valset = torch.utils.data.random_split(train_and_validation_set, [int(0.95*len(train_and_validation_set)), len(train_and_validation_set)-int(0.95*len(train_and_validation_set))])

# x = [0,0,0,0,0,0,0,0,0,0]
# print(len(x))
# for i in range(45000):
#     x[trainset.__getitem__(i)[1]] += 1

# for i in range(5000):
#     x[valset.__getitem__(i)[1]] += 1
# print (x)



lr = 0.02
initial_gamma = 0.1
neural_network = []
neural_network.append(Linear_layer(3*32*32,1000,lr))
neural_network.append(Leaky_ReLU(initial_gamma,lr))
#neural_network.append(ReLU())


neural_network.append(Linear_layer(1000,500,lr))
neural_network.append(Leaky_ReLU(initial_gamma,lr))
#neural_network.append(ReLU())

neural_network.append(Linear_layer(500,200,lr))
neural_network.append(Leaky_ReLU(initial_gamma,lr))
#neural_network.append(ReLU())

neural_network.append(Linear_layer(200,100,lr))
neural_network.append(Leaky_ReLU(initial_gamma,lr))
#neural_network.append(ReLU())

neural_network.append(Linear_layer(100,50,lr))
neural_network.append(Leaky_ReLU(initial_gamma,lr))
#neural_network.append(ReLU())

neural_network.append(Linear_layer(50,20,lr))
neural_network.append(Leaky_ReLU(initial_gamma,lr))
#neural_network.append(ReLU())

neural_network.append(Linear_layer(20,10,lr))


train_acc = []
val_acc = []
test_acc = []
mean_cost = []
gamma_list = []
gamma_list.append(initial_gamma)
size_of_batch_train = 128
size_of_batch_val = 100
size_of_batch_test = 100
number_of_epoch = 200
for epoch in range(number_of_epoch):
    i = 0
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=size_of_batch_train, shuffle=True, num_workers=0)
    for train_data in trainloader:
        train_images, train_labels = train_data
        train_input = train_images.view(-1,3*32*32).numpy()
        train_output = train_labels.numpy()
        train(neural_network,train_input,train_output)

        i = i + 1
        if i%int(len(trainset)/(size_of_batch_train*4)) == 0:
            print('Epoch No.%d training %f finished' %(epoch+1, i*size_of_batch_train/len(trainset)))
            #dloss_over_dinput_of_softmax = (calculate_dmean_cost_crossentropy_over_dinputofSoftmax(forward(neural_network,train_input)[-1], train_output))
            #print(dloss_over_dinput_of_softmax[0])
            #dloss_over_dinput_of_linear = (neural_network[-1].backward(forward(neural_network, train_input)[-2],dloss_over_dinput_of_softmax))
            #print(dloss_over_dinput_of_linear[0])
            
            #dloss_over_dinput_of_relu = (neural_network[-2].backward(forward(neural_network, train_input)[-3],dloss_over_dinput_of_linear))
            #print(dloss_over_dinput_of_relu[0])


            #print((forward(neural_network, train_input)[0])[0])
            #print((forward(neural_network, train_input)[1])[0])
            #print((neural_network[0].forward(train_input))[0])
            #print(neural_network[1].gamma)
    if isinstance(neural_network[1], Leaky_ReLU):
        gamma_list.append(neural_network[1].gamma)


    total_cost = 0
    train_correct = 0
    train_total = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=size_of_batch_train, shuffle=False, num_workers=0)
    for train_data in trainloader:
        train_images, train_labels = train_data
        train_input = train_images.view(-1,3*32*32).numpy()
        train_output = train_labels.numpy()
        train_correct += (np.sum(predict(neural_network, train_input) == train_output))
        total_cost += size_of_batch_train * mean_cost_crossentropy(forward(neural_network, train_input)[-1], train_output)
    mean_cost.append(total_cost/len(trainset))
    train_acc.append(train_correct/train_total)

    val_correct = 0
    val_total = len(valset)
    valloader = torch.utils.data.DataLoader(valset, batch_size=size_of_batch_val, shuffle=False, num_workers=0)
    for val_data in valloader:
        val_images, val_labels = val_data
        val_input = val_images.view(-1,3*32*32).numpy()
        val_output = val_labels.numpy()
        val_correct += (np.sum(predict(neural_network, val_input) == val_output))
    
    val_acc.append(val_correct/val_total)

    test_correct = 0
    test_total = len(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=size_of_batch_test, shuffle=False, num_workers=0)
    for test_data in testloader:
        test_images, test_labels = test_data
        test_input = test_images.view(-1,3*32*32).numpy()
        test_output = test_labels.numpy()
        test_correct += (np.sum(predict(neural_network, test_input) == test_output))
 
    test_acc.append(test_correct/test_total)

print(gamma_list)
print(mean_cost)
print(train_acc)
print(val_acc)
print(test_acc)



plt.figure()
plt.plot([i+1 for i in range(number_of_epoch)], train_acc,marker='*', label='train accuracy', color = 'blue')
plt.plot([i+1 for i in range(number_of_epoch)], val_acc, marker='o',label='validation accuracy', color = 'black')

#plt.plot([i+1 for i in range(number_of_epoch)], test_acc,marker = 'v',label='test accuracy', color = 'red')
plt.grid()
plt.legend(loc = 'lower right')
plt.yticks(np.arange(0,1.1,0.1))
title1 = 'Learning Rate: ' + str(lr) + '  Batch Size: ' + str(size_of_batch_train)
plt.title(title1)
plt.xlabel('Training Epoch')
plt.ylabel('Accuracy')



plt.figure()
plt.plot([i+1 for i in range(number_of_epoch)], mean_cost)
plt.grid()
plt.xlabel('Training Epoch')
plt.ylabel('Mean Cost')

if isinstance(neural_network[1], Leaky_ReLU):
    plt.figure()
    plt.plot(gamma_list)
    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Learnable Parameter Gamma')
plt.show()



# testloader = torch.utils.data.DataLoader(testset, batch_size=size_of_batch_test, shuffle=False, num_workers=0)
# test_input = []
# test_output = []
# for test_data in testloader:
#     test_images, test_labels = test_data
#     test_input = test_input + test_images.view(-1,3*32*32).numpy().tolist()
#     test_output = test_output + test_labels.numpy().tolist()
# test_input = np.array(test_input)
# test_output = np.array(test_output)
# print(np.mean(predict(neural_network,test_input)==test_output))



