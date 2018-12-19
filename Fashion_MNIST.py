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
testloader = t.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

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
epochs = 1

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

#testing of model
#test_img, test_lbl = next(iter(testset))
for test_img, test_lbl in trainloader:
    test_img += test_img
img = test_img[0].view(1,784)
with t.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim = 1)

helper.view_classify(img.view(1,28,28), ps, version = 'Fashion')

#testing of model's accuracy
for test_img2, test_lbl2 in testloader:
    test_img2 += test_img2
#print(len(image))
acc_test_img2 = test_img2.view(test_img2.shape[0], -1)
with t.no_grad():
    logits2 = model.forward(acc_test_img2)

ps2 = F.softmax(logits2, dim = 1)
print(ps2.shape)


#measuring accuracy
top_p, top_class = ps2.topk(1, dim = 1)

equals = top_class == test_lbl2.view(*top_class.shape)

print('top class = ', top_class)
print('test lebel = ', test_lbl2.view(top_class.shape))

print('equal', equals)

#accuracy
accuracy = t.mean(equals.type(t.FloatTensor))
print(f'accuracy: {accuracy.item()*100}%')