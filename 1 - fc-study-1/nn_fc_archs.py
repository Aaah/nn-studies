import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import pickle
from iatools import cli_progress_test, pick_nsamples, csv_stats, csv_stats_ratio, split_dataset
from numpy import array, ceil, arange
import csv

# -- NET DEFINITION
class NN_FC(nn.Module):
    """Fully-connected only NN. If the structure of the layers is None, layers are randomnly generated"""
    def __init__(self, nclasses=2, nlayers=1, nneurons=32, ninputs=128, layers=None):
        super(NN_FC, self).__init__()
        
        self.nlayers = nlayers
        self.nclasses = nclasses
        self.nneurons = nneurons
        self.ninputs = ninputs

        # -- compute the number of neurons in every layer
        if layers == None:
            self.layers = []
            for l in range(nlayers-1):
                self.layers.append(randint(1, nneurons - sum(self.layers) - nlayers))

            self.layers.append(nneurons - sum(self.layers))
            self.layers.append(nclasses)
        else:
            self.layers = list(layers)
            self.nneurons = sum(self.layers)
            self.nlayers = len(self.layers)
            self.layers.append(nclasses)

        # -- create fully connected layers
        modules = []
        modules.append(nn.Linear(ninputs, self.layers[0]))
        for l in range(nlayers):
            modules.append(nn.Linear(self.layers[l], self.layers[l+1]))
            if l < nlayers:
                modules.append(nn.ReLU())
        
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# -- TRAINING LOOP
do_training = True

DATASET = 'data-n3.pkl'
TESTPROPORTION = 0.2
NLAYERS = 2
NNEURONS = 128
NMODELS = 2
NSTATS = 25
NEPOCHS = 25
NBATCHS = 500
MINIBATCHSIZE = 50
NINPUTS = 128
NCLASSES = 2
MODELSTEP = 1

csv_filename = './fc_%d_%d_SHUFFLE.csv' % (NLAYERS, NNEURONS)
# csv_filename = './DUMP.csv'

if do_training:

    for m in arange(1, NMODELS, MODELSTEP):

        _first_layer_n  = 32
        _second_layer_n = NNEURONS - _first_layer_n

        for s in range(NSTATS):

            # -- load data
            X, Y, _ = pickle.load(open(DATASET,'rb'))

            # -- split dataset into training and testing dataset
            train_idx, test_idx = split_dataset(X, 1.0 - TESTPROPORTION)
            X_train, Y_train    = X[train_idx], Y[train_idx]
            X_test, Y_test      = X[test_idx], Y[test_idx]
            
            # -- instanciate nn (convert all weights to single precision floats)
            net = NN_FC(nclasses    = NCLASSES,                           # classification classes 
                        nlayers     = NLAYERS,                            # number of layers in the nn
                        nneurons    = NNEURONS,                           # total number of neurons
                        ninputs     = NINPUTS,                            # input dimension
                        layers      = (_first_layer_n, _second_layer_n)   # opt : structure of the nn
                        ).float()

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

            # TODO : save model if better than previous
            # TODO : normalize inputs
            # TODO : regarder la sortie des couches intermédiaires

            # # -- SAVE MODEL
            # torch.save(net.state_dict(), MODEL_PATH)

        print()

# csv_stats(csv_filename)
# csv_stats_ratio(csv_filename)