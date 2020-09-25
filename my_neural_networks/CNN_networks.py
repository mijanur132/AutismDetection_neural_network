import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

outputClassN=2
imageRes=224

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.Var1=2*2*3
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=14, stride=1, padding=7)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=14, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=21, stride=7, padding=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=6,stride=3,padding=1)
        self.fc1 = nn.Linear(self.Var1, 10)
        self.fc2 = nn.Linear(10, outputClassN)

    def forward(self, x):
        # 2D convolutional layer with max pooling layer and reLU activations
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 2nd layer of 2D convolutional layer with max pooling layer and reLU activations
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # rearrange output from a 2d array (matrix) to a vector
        x = x.view(-1, self.Var1)
        # fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # fully connected layer
        x = self.fc2(x)
        # log(softmax(x)) activation
        return F.log_softmax(x, dim=1)


class Net2():
    def __init__(self,gpu_id=-1):

        if(gpu_id!=-1):
            self.model = CNNNet().cuda(gpu_id)
        else:
            self.model = CNNNet()


    def train_one_batch(self, X, y, y_1hot,imageRes):
        """Train for one batch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        printout=False
        x = X.view(-1, 1, imageRes,imageRes)
        self.model.train()
        # Forward pass. Gets output *before* softmax
        output = self.model.forward(x)

        ys=torch.squeeze(y)
        # Computes the negative log-likelihood over the training data target.
        #     log(softmax(.)) avoids computing the normalization factor.
        #     Note that target does not need to be 1-hot encoded (pytorch will 1-hot encode for you)

        ys=ys.type(torch.LongTensor)
        loss = F.nll_loss(output, ys)

        # **Very important,** need to zero the gradient buffer before we can use the computed gradients
        self.model.zero_grad()  # zero the gradient buffers of all parameters

        if printout:
            print('conv1.bias.grad before backward')
            print(self.model.conv1.bias.grad)

        # Backpropagate loss for all paramters over all examples
        loss.backward()

        if printout:
            print('conv1.bias.grad after backward')
            print(self.model.conv1.bias.grad)

            # Performs one step of SGD with a fixed learning rate (not a Robbins-Monro procedure)
        learning_rate = 1e-3

        if printout:
            print("")
            print("conv1.bias before SGD step")
            print(self.model.conv1.bias)

        # iterate over all model parameters
        for f in self.model.parameters():
            # why subtract? Are we minimizing or maximizing?
            f.data.sub_(f.grad.data * learning_rate)

        if printout:
            print("conv1.bias after SGD step")
            print(self.model.conv1.bias)

        if printout:
            print("")
            print("... next gradient step")
        return loss

    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(X.view(-1,1,imageRes,imageRes))
            ys = torch.squeeze(y)
            ys = ys.type(torch.LongTensor)
            loss = F.nll_loss(outputs, ys)
            return loss

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(X.view(-1,1,imageRes,imageRes))
            return torch.max(outputs,1)[1]




