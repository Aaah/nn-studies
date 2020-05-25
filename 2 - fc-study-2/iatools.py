import pickle
import matplotlib.pyplot as plt
import random
import sys
import csv
from numpy import array, linspace, zeros, mean, std, min, concatenate, argsort, flipud, argmax, arange

def data_normalize_1(data):
    """Center data and bound amplitude in [-1; +1]"""
    for i, r in enumerate(data):
        m = mean(r)
        r -= m
        r = r / max(abs(r))
        data[i] = r
    
    return data

def data_normalize_2(data):
    """Center data and normalize standard deviation"""
    for i, r in enumerate(data):
        m = mean(r)
        r -= m
        r = r / std(r)
        data[i] = r
    return data

def data_normalize_3(data):
    """Shuffle data randomly"""
    idx = arange(len(data[0]))
    random.shuffle(idx)
    for i, r in enumerate(data):
        data[i] = r[idx]

    return data

def cli_progress_test(i, end_val, bar_length=20, prefix=""):
    percent = float(i) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\r{:<50} {} {}%".format(prefix, hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def pick_nsamples(data, nsamples=1):    
    idx = random.sample(range(data.shape[0]), nsamples)
    return data[idx,:], idx

def split_dataset(data, ratio=0.9):
    full_idx = range(data.shape[0])
    idx = sorted(random.sample(full_idx, int(data.shape[0] * ratio)))
    cidx = [e for e in full_idx if e not in idx] 
    return  idx, cidx

def shuffle_dataset(data):
    idx = random.sample(range(data.shape[0]), data.shape[0])
    return  idx

def csv_get_every_instance(csv_filename):
    """
    - INPUT : csv file
    - OUTPUT : arrays of each convergence (epochs) for all stats, learning and testing
    - Expected structure CSV FILE:
       - c1 : model number
       - c2 : number of parameters in the model
       - c3 : statistic iteration
       - c4 : epoch count
       - c5 : number of inputs of the NN
       - c6 : number of neurons in the first layer..
       - c7 : number ...
       - c[-1] : accuracy on test dataset
       - c[-2] : loss on training dataset
       - c[-3] : number of neurons in the output layer (number of classes in the classification)
    """

    data = []
    with open(csv_filename) as f:
        r = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in r:
            data.append(row)

    data = array(data)

    # -- dimensions
    nepochs = int(max(data.T[3]) + 1)
    mdls = []; [mdls.append(int(x)) for x in data.T[0] if x not in mdls] 
    nmodels = int(len(mdls))
    nstats = int(max(data.T[2]) + 1)

    print(nepochs, mdls, nmodels, nstats)

    # -- iterate over each stat
    learn_d = zeros((nstats, nepochs))
    test_d = zeros((nstats, nepochs))
    for r in data:
        learn_d[int(r[2]), int(r[3])] = r[-2]
        test_d[int(r[2]), int(r[3])] = r[-1]

    return learn_d, test_d

def csv_stats_convergence(csv_filename):
    """Expected structure CSV FILE:
       - c1 : model number
       - c2 : number of parameters in the model
       - c3 : statistic iteration
       - c4 : epoch count
       - c5 : number of inputs of the NN
       - c6 : number of neurons in the first layer..
       - c7 : number ...
       - c[-1] : accuracy on test dataset
       - c[-2] : loss on training dataset
       - c[-3] : number of neurons in the output layer (number of classes in the classification)
    """

    data = []
    with open(csv_filename) as f:
        r = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in r:
            data.append(row)

    data = array(data)

    # -- number of epochs to compute stats
    nepochs = int(max(data.T[3]) + 1)
    mdls = []; [mdls.append(int(x)) for x in data.T[0] if x not in mdls] 
    nmodels = int(len(mdls))
    nstats = int(max(data.T[2]) + 1)

    # -- find the model that has the best score in the last epoch
    test_d = []
    for r in data:
        if r[3] == nepochs-1:
            test_d.append(concatenate([[r[0]], [r[-1]]]))

    test_d = array(test_d)
    
    accuracy_esp = zeros(nmodels)
    for i, m in enumerate(mdls):
        _tmp = test_d[test_d[:,0] == m][:,1]
        accuracy_esp[i] = mean(_tmp)

    mdl_idx = argmax(accuracy_esp)

    # -- extract all epochs and stats of the model found
    test_d = []; learn_d = []
    for r in data:
        if r[0] == mdl_idx:
            test_d.append([r[3], r[-1]])
            learn_d.append([r[3], r[-2]])

    test_d = array(test_d); learn_d = array(learn_d)

    # -- compute statistics for all epochs
    acc_esp = zeros(nepochs)
    acc_std = zeros(nepochs)
    loss_std = zeros(nepochs)
    loss_esp = zeros(nepochs)
    for i, m in enumerate(range(nepochs)):
        _tmp = test_d[test_d[:,0] == m][:,1]
        acc_esp[i] = mean(_tmp)
        acc_std[i] = std(_tmp)

        _tmp = learn_d[learn_d[:,0] == m][:,1]
        loss_esp[i] = mean(_tmp)
        loss_std[i] = std(_tmp)

    return loss_esp, loss_std, acc_esp, acc_std
