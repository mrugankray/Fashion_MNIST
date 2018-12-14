# imports
import torch as t
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import optim
import helper

#defining transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download= True, train= True, transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#test set
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download= True, train= False, transform=transform)
trainloader = t.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#image, label = next(iter(trainloader))
#helper.imshow(image[0,:])

#flattening the images
#image = image.view(image.shape[0], -1)

#creating model
model = nn.Sequential(nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim = 1))

#loss function
criterion = nn.NLLLoss()

#optimmizer
optimizer = optim.Adam(model.parameters(), lr= 0.003)

#epochs
epochs = 5

for e in range(epochs):
    running_loss = 0
    for image, label in trainloader:
    
        #flattening the images
        image = image.view(image.shape[0], -1)

        #setting all grads to zero
        optimizer.zero_grad()

        #probability(output)
        #forward propagation
        log_pb = model.forward(image)

        #calculate loss
        loss = criterion(log_pb, label)

        #backward propagation
        loss.backward()

        #adams will start working
        optimizer.step()

        running_loss = running_loss+loss.item()
        #print(loss)
        print('loss.item() = ', loss.item())
        print('running_loss = ', running_loss)
    else:
        print(f"training loss: {running_loss/len(trainloader)}")

img = image[0].view(1,784)
with t.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim = 1)

helper.view_classify(img.view(1,28,28), ps, version = 'Fashion')