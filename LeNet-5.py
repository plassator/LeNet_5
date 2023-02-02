import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

class LeNet_5(nn.Module):
    def __init__(self) -> None:
        super(LeNet_5,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5,stride=1,padding=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(6,16,5,1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(16,120,5,1),     
        )

        self.fc = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10), 
        )

    def forward(self,input):
        x = self.conv(input)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = nn.Softmax(dim=1)(x)
        return x 
        

def read_data():
    prepare = transforms.ToTensor()
    train_data = MNIST(root='./train',train=True,download=True,transform=prepare)
    test_data = MNIST(root='./test',train=False,download=True,transform=prepare)

    return train_data,test_data

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
   
    model.train()

    average_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()

        average_loss += loss

        if batch % 1000 == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")       
    
    return (average_loss/size)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(
			torch.float).sum().item()

    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # why tensor here?
    return test_loss.item()

def train_cnn(train_data,test_data,epoch):

    model = LeNet_5().to(device)
    optimizer = opt.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.NLLLoss()

    BATCH_SIZE=16
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE)
    
    test_losses,train_losses = [],[]
    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses.append(train(trainloader, model, loss_fn, optimizer))
        test_losses.append(test(testloader, model, loss_fn))
  
    torch.save(model.state_dict(),'LeNet_5_state_dict.pth')  

    return test_losses,train_losses

def save_curve(test_losses,train_losses,epoch):

    x = [i for i in range(epoch)]

    fig,ax = plt.subplots()
    ax.plot(x,train_losses,label='training')
    ax.plot(x,test_losses,label='test')
    ax.set_title('Learning Curve')
    ax.set_ylim(top=1.)
    plt.xticks(np.arange(0,epoch+1,5))
    plt.legend(loc='upper right')
    plt.savefig('learning_curve.jpg')

if __name__ == "__main__":
    epoch = 4

    train_data,test_data = read_data()
    test_losses,train_losses = train_cnn(train_data,test_data,epoch)
    save_curve(test_losses,train_losses,epoch)
       