import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import pickle
from iatools import cli_progress_test, pick_nsamples, split_dataset
from numpy import array, ceil, arange
import csv

# -- NET DEFINITION
class NN_FC_LEAKY(nn.Module):
    """Fully-connected only NN."""
    def __init__(self, layers=None):
        super(NN_FC_LEAKY, self).__init__()
        
        self.layers = layers

        L = len(layers)-1
        modules = []
        for l in range(L):
            modules.append(nn.Linear(layers[l], layers[l+1]))
            if l < L-1:
                modules.append(nn.LeakyReLU())
                # modules.append(nn.LeakyReLU(negative_slope=5e-1))
        
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NN_FC(nn.Module):
    """Fully-connected only NN."""
    def __init__(self, layers=None):
        super(NN_FC, self).__init__()
        
        self.layers = layers

        L = len(layers)-1
        modules = []
        for l in range(L):
            modules.append(nn.Linear(layers[l], layers[l+1]))
            if l < L-1:
                modules.append(nn.ReLU())
        
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    
    # -- TRAINING LOOP
    do_training = True

    DATASET = './DATA/2d-data-pounds.pkl'
    TESTPROPORTION = 0.2
    NLAYERS = 2
    NNEURONS = 25
    NMODELS = 2
    NSTATS = 5
    NEPOCHS = 100
    NBATCHS = 100
    MINIBATCHSIZE = 25
    NINPUTS = 2
    NCLASSES = 2
    MODELSTEP = 1

    csv_filename = './TEMP/fcnn_%d_%d_DUMP.csv' % (NLAYERS, NNEURONS)
    model_filename = './TEMP/model_%d_%d_BEST.mdl' % (NLAYERS, NNEURONS)

    if do_training:

        for m in arange(1, NMODELS, MODELSTEP):

            best_accuracy = 0.0

            for s in range(NSTATS):

                # -- load data
                X, Y = pickle.load(open(DATASET,'rb'))

                # -- split dataset into training and testing dataset
                train_idx, test_idx = split_dataset(X, 1.0 - TESTPROPORTION)
                X_train, Y_train    = X[train_idx], Y[train_idx]
                X_test, Y_test      = X[test_idx], Y[test_idx]
                
                # -- instanciate nn (convert all weights to single precision floats)
                net = NN_FC_LEAKY(layers = [NINPUTS, NNEURONS, NNEURONS, NCLASSES]).float()

                # -- optimisation criteration
                criterion = nn.CrossEntropyLoss()

                # -- optimiser
                optimizer = optim.Adam(net.parameters(), lr=0.001)
                    
                # -- TRAIN
                for epoch in range(NEPOCHS):  
                    running_loss = 0.0
                    for i in range(1, NBATCHS):

                        # get minibatch of data and labels
                        data, idx = pick_nsamples(X_train, MINIBATCHSIZE)
                        inputs, labels = torch.tensor(data).float(), torch.tensor(Y_train[idx]).float()

                        # process input image through network
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())

                        # backprop and optimisation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    # -- TEST
                    correct = 0.0; total = 0.0
                    with torch.no_grad():
                        for i in range(1, int(ceil(X_test.shape[0] / MINIBATCHSIZE))):
                            idx = range(i * MINIBATCHSIZE, (i+1) * MINIBATCHSIZE)
                            _inputs, _groundtruth = torch.tensor(X_test[idx]).float(), torch.tensor(Y_test[idx]).float()
                            outputs = net(_inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += MINIBATCHSIZE
                            correct += (predicted == _groundtruth).sum().item()

                    # -- print statistics
                    train_loss = round(running_loss / NBATCHS,3)
                    test_accuracy = round(100.0 * correct / total,1)

                    model_str = '%3d' % NINPUTS
                    for e in net.layers:
                        model_str += ', %3d' % e

                    prefix_str = "[model: %s (%d params)] [stats: %d/%d] [epoch: %02d/%02d] loss = %.3f - test accuracy = %0.1f %%" % (model_str, net.count_parameters(), s, NSTATS, epoch+1, NEPOCHS, train_loss, test_accuracy)
                    cli_progress_test(s * NEPOCHS + (epoch + 1), NSTATS * NEPOCHS, bar_length=20, prefix=prefix_str)

                    # -- export results
                    with open(csv_filename, 'a+') as f:
                        w = csv.writer(f)
                        row = []
                        row.append(m)       # model
                        row.append(net.count_parameters())
                        row.append(s)       # stats
                        row.append(epoch)   # epoch

                        # - network shape
                        row.append(NINPUTS)
                        for e in net.layers:
                            row.append(e)

                        # - results
                        row.append(train_loss) # training 
                        row.append(test_accuracy)    
                        w.writerow(row)

                        # - export model
                        if test_accuracy == max(test_accuracy, best_accuracy):
                            best_accuracy = test_accuracy
                            torch.save(net.state_dict(), model_filename)

            print()
