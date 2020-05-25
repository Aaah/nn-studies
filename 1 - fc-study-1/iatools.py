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

def csv_stats(csv_filename):
    
    data = []
    with open(csv_filename) as f:
        r = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in r:
            data.append(row)

    data = array(data)

    # -- number of epochs to compute stats
    nepochs = 1
    nmodels = 0
    for r in data:
        nepochs = int(max(nepochs, r[0] + 1))
        if r[0] == 0:
            nmodels += 1 

    vals = zeros((nepochs,nmodels))
    for i, r in enumerate(data):
        vals[int(r[0]), int(i / nepochs)] = r[-1]

    means = mean(vals, axis=1)
    stds = std(vals, axis=1)

    plt.figure(figsize=(8,4))
    x = range(1, nepochs + 1)
    plt.plot(x, means, 'r')
    plt.plot(x, means + stds, 'b--', lw=1)
    plt.plot(x, means - stds, 'b--', lw=1)
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.ylim([-0.1,1.1])
    plt.tight_layout()
    plt.show()

    # todo : compute mean and std for all epochs and display

    return

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

def csv_stat_cv_1model(csv_filename):
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

    # 1. 1 model
    # 2. compute stats for all epochs

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

    # -- extract all epochs and stats of the model found
    test_d = []; learn_d = []
    for r in data:
        # if r[0] == mdl_idx:
        test_d.append([r[3], r[-1]])
        learn_d.append([r[3], r[-2]])

    test_d = array(test_d); learn_d = array(learn_d)

    # -- compute statistics for all epochs
    acc_esp = zeros(nepochs)
    acc_min = zeros(nepochs)
    acc_max = zeros(nepochs)
    acc_std = zeros(nepochs)
    loss_std = zeros(nepochs)
    loss_esp = zeros(nepochs)
    loss_min = zeros(nepochs)
    loss_max = zeros(nepochs)
    for i, m in enumerate(range(nepochs)):
        _tmp = test_d[test_d[:,0] == m][:,1]
        acc_esp[i] = mean(_tmp)
        acc_min[i] = min(_tmp)
        acc_max[i] = max(_tmp)
        acc_std[i] = std(_tmp)

        _tmp = learn_d[learn_d[:,0] == m][:,1]
        loss_esp[i] = mean(_tmp)
        loss_std[i] = std(_tmp)
        loss_min[i] = min(_tmp)
        loss_max[i] = max(_tmp)

    return loss_esp, loss_std, loss_min, loss_max, acc_esp, acc_std, acc_min, acc_max

def csv_stats_ratio(csv_filename):
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

    # -- number of parameters
    nparams = []; [nparams.append(int(x)) for x in data.T[1] if x not in nparams] 

    # -- remove all epochs but the last one
    learn_d = []
    test_d = []
    for r in data:
        if r[3] == nepochs-1:
            test_d.append(concatenate([[r[0]], [r[-1]]]))
            learn_d.append(concatenate([[r[0]], [r[-2]]]))

    test_d = array(test_d)
    learn_d = array(learn_d)

    # -- compute statistics
    loss_esp = zeros(nmodels);
    loss_min = zeros(nmodels);
    loss_std = zeros(nmodels);
    accuracy_esp = zeros(nmodels);
    accuracy_min = zeros(nmodels);
    accuracy_std = zeros(nmodels);
    ratio = linspace(0.0, 1.0, nmodels+1, endpoint=False)[1:]

    for i, m in enumerate(mdls):
        _tmp = learn_d[learn_d[:,0] == m][:,1]
        loss_esp[i] = mean(_tmp)
        loss_std[i] = std(_tmp)
        loss_min[i] = min(_tmp)
        
        _tmp = test_d[test_d[:,0] == m][:,1]
        accuracy_esp[i] = mean(_tmp)
        accuracy_std[i] = std(_tmp)
        accuracy_min[i] = min(_tmp)

    # plt.figure(figsize=(8,4))
    # plt.subplot(211)
    # plt.plot(ratio, loss_esp, 'bx')
    # plt.plot(ratio, loss_min, 'rx')
    # plt.plot(ratio, loss_esp + loss_std, 'b--', lw=1)
    # plt.plot(ratio, loss_esp - loss_std, 'b--', lw=1)
    # plt.ylabel("LOSS")
    # plt.xlabel("Ratio L1 / L2")
    # plt.ylim([0.0, 1.0])
    # plt.subplot(212)
    # plt.plot(ratio, accuracy_esp, 'bx')
    # plt.plot(ratio, accuracy_min, 'rx')
    # plt.plot(ratio, accuracy_esp + accuracy_std, 'b--', lw=1)
    # plt.plot(ratio, accuracy_esp - accuracy_std, 'b--', lw=1)
    # plt.ylabel("ACCURACY")
    # plt.xlabel("Ratio L1 / L2")
    # plt.ylim([0.0, 100.0])
    # plt.tight_layout()
    # plt.show()

    return loss_esp, loss_std, loss_min, accuracy_esp, accuracy_std, accuracy_min, nparams

if __name__ == "__main__":

    LOAD = False
    NORM_DATA1 = False
    NORM_DATA2 = False
    NORM_DATA3 = False
    DISPLAY_STATS_RATIO = True
    DISPLAY_CONVERGENCE = False
    DISPLAY_PERFS_DATABASE = False

    if NORM_DATA1:
        # -- load data
        X, Y, sigmas = pickle.load(open('data.pkl','rb'))

        # -- normalize data
        newX = data_normalize_1(X)
        pickle.dump((newX, Y, sigmas), open('data-n1.pkl','wb'))

        # -- pick random samples from the data
        samples, idx = pick_nsamples(X, 10)
        samples0 = samples[Y[idx] == 0, :]
        samples1 = samples[Y[idx] == 1, :]

        # -- display data
        plt.figure(figsize=(12,6))

        # display samples from class 0
        plt.subplot(211)
        plt.title("Class 0 samples")
        plt.plot(samples0.T, lw=1)

        # display samples from class 1
        plt.subplot(212)
        plt.title("Class 1 samples")
        plt.plot(samples1.T, lw=1)

        plt.tight_layout()
        plt.show()

    if NORM_DATA2:
        # -- load data
        X, Y, sigmas = pickle.load(open('data.pkl','rb'))

        # -- normalize data
        newX = data_normalize_2(X)
        pickle.dump((newX, Y, sigmas), open('data-n2.pkl','wb'))

        # -- pick random samples from the data
        samples, idx = pick_nsamples(X, 10)
        samples0 = samples[Y[idx] == 0, :]
        samples1 = samples[Y[idx] == 1, :]

        # -- display data
        plt.figure(figsize=(12,6))

        # display samples from class 0
        plt.subplot(211)
        plt.title("Class 0 samples")
        plt.plot(samples0.T, lw=1)

        # display samples from class 1
        plt.subplot(212)
        plt.title("Class 1 samples")
        plt.plot(samples1.T, lw=1)

        plt.tight_layout()
        plt.show()

    if NORM_DATA3:
        # -- load data
        X, Y, sigmas = pickle.load(open('data.pkl','rb'))

        # -- normalize data
        newX = data_normalize_3(X)
        pickle.dump((newX, Y, sigmas), open('data-n3.pkl','wb'))

        # -- pick random samples from the data
        samples, idx = pick_nsamples(X, 10)
        samples0 = samples[Y[idx] == 0, :]
        samples1 = samples[Y[idx] == 1, :]

        # -- display data
        plt.figure(figsize=(12,6))

        # display samples from class 0
        plt.subplot(211)
        plt.title("Class 0 samples")
        plt.plot(samples0.T, lw=1)

        # display samples from class 1
        plt.subplot(212)
        plt.title("Class 1 samples")
        plt.plot(samples1.T, lw=1)

        plt.tight_layout()
        plt.show()

    if DISPLAY_CONVERGENCE:
        loss_mean_128, loss_std_128, acc_mean_128, acc_std_128 = csv_stats_convergence("./results_fc/fc_arch_results_2_128_FULL_ARCH.csv")
        loss_mean_64, loss_std_64, acc_mean_64, acc_std_64 = csv_stats_convergence("./results_fc/fc_arch_results_2_64_FULL_ARCH.csv")
        loss_mean_32, loss_std_32, acc_mean_32, acc_std_32 = csv_stats_convergence("./results_fc/fc_arch_results_2_32_FULL_ARCH.csv")
        loss_mean_16, loss_std_16, acc_mean_16, acc_std_16 = csv_stats_convergence("./results_fc/fc_arch_results_2_16_FULL_ARCH.csv")
        loss_mean_8, loss_std_8, acc_mean_8, acc_std_8 = csv_stats_convergence("./results_fc/fc_arch_results_2_8_FULL_ARCH.csv")

        plt.figure(figsize=(12,5))

        plt.subplot(121)

        # -- LOSS
        N = len(loss_mean_8)
        plt.plot(linspace(0.0,N,N), loss_mean_8, 'g', label="8")
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_8 + loss_std_8, flipud(loss_mean_8 - loss_std_8)]), color=(0.0,1.0,0.0,0.1))

        N = len(loss_mean_16)
        plt.plot(linspace(0.0,N,N), loss_mean_16, 'r', label="16")
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_16 + loss_std_16, flipud(loss_mean_16 - loss_std_16)]), color=(1.0,0.0,0.0,0.1))

        N = len(loss_mean_32)
        plt.plot(linspace(0.0,N,N), loss_mean_32, 'k', label="32")
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_32 + loss_std_32, flipud(loss_mean_32 - loss_std_32)]), color=(0.0,0.0,0.0,0.1))

        N = len(loss_mean_64)
        plt.plot(linspace(0.0,N,N), loss_mean_64, label="64", color=(0.0,1.0,1.0,1.0))
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_64 + loss_std_64, flipud(loss_mean_64 - loss_std_64)]), color=(0.0,1.0,1.0,0.1))

        N = len(loss_mean_128)
        plt.plot(linspace(0.0,N,N), loss_mean_128, label="128", color=(1.0,0.0,1.0,1.0))
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_128 + loss_std_128, flipud(loss_mean_128 - loss_std_128)]), color=(1.0,0.0,1.0,0.1))

        plt.xlabel("EPOCHS")
        plt.ylabel("LOSS (LEARNING)")
        plt.tight_layout()
        plt.legend()
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.ylim([-0.1,1.1])

        # -- ACCURACY
        plt.subplot(122)

        N = len(acc_mean_8)
        plt.plot(linspace(0.0,N,N), acc_mean_8, 'g', label="8")
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_8 + acc_std_8, flipud(acc_mean_8 - acc_std_8)]), color=(0.0,1.0,0.0,0.1))

        N = len(acc_mean_16)
        plt.plot(linspace(0.0,N,N), acc_mean_16, 'r', label="16")
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_16 + acc_std_16, flipud(acc_mean_16 - acc_std_16)]), color=(1.0,0.0,0.0,0.1))

        N = len(acc_mean_32)
        plt.plot(linspace(0.0,N,N), acc_mean_32, 'k', label="32")
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_32 + acc_std_32, flipud(acc_mean_32 - acc_std_32)]), color=(0.0,0.0,0.0,0.1))

        N = len(acc_mean_64)
        plt.plot(linspace(0.0,N,N), acc_mean_64, label="64", color=(0.0,1.0,1.0,1.0))
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_64 + acc_std_64, flipud(acc_mean_64 - acc_std_64)]), color=(0.0,1.0,1.0,0.1))

        N = len(acc_mean_128)
        plt.plot(linspace(0.0,N,N), acc_mean_128, label="128", color=(1.0,0.0,1.0,1.0))
        x = linspace(0.0,N,N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_128 + acc_std_128, flipud(acc_mean_128 - acc_std_128)]), color=(1.0,0.0,1.0,0.1))

        plt.xlabel("EPOCHS")
        plt.ylabel("ACCURACY (TEST)")
        plt.tight_layout()
        plt.legend()
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.ylim([50.0,100.0])

        plt.show()


    if DISPLAY_STATS_RATIO:
        loss_mean_4, loss_std_4, loss_min_4, acc_mean_4, acc_std_4, acc_min_4, np4                  = csv_stats_ratio("./results_fc/CONVERGENCE/fc_arch_results_2_4_FULL_ARCH.csv")
        loss_mean_8, loss_std_8, loss_min_8, acc_mean_8, acc_std_8, acc_min_8, np8                  = csv_stats_ratio("./results_fc/CONVERGENCE/fc_arch_results_2_8_FULL_ARCH.csv")
        loss_mean_16, loss_std_16, loss_min_16, acc_mean_16, acc_std_16, acc_min_16, np16           = csv_stats_ratio("./results_fc/CONVERGENCE/fc_arch_results_2_16_FULL_ARCH.csv")
        loss_mean_32, loss_std_32, loss_min_32, acc_mean_32, acc_std_32, acc_min_32, np32           = csv_stats_ratio("./results_fc/CONVERGENCE/fc_arch_results_2_32_FULL_ARCH.csv")
        loss_mean_64, loss_std_64, loss_min_64, acc_mean_64, acc_std_64, acc_min_64, np64           = csv_stats_ratio("./results_fc/CONVERGENCE/fc_arch_results_2_64_FULL_ARCH.csv")
        loss_mean_128, loss_std_128, loss_min_128, acc_mean_128, acc_std_128, acc_min_128, np128    = csv_stats_ratio("./results_fc/CONVERGENCE/fc_arch_results_2_128_FULL_ARCH.csv")

        plt.figure(figsize=(15,5))
        
        # -- LOSS
        plt.subplot(131)

        N = len(loss_mean_4)
        plt.plot(linspace(0.0, 1.0, N), loss_mean_4, 'b', label="4")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_4 + loss_std_4, flipud(loss_mean_4 - loss_std_4)]), color=(0,0,1.0,0.1))

        N = len(loss_mean_8)
        plt.plot(linspace(0.0, 1.0, N), loss_mean_8, 'g', label="8")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_8 + loss_std_8, flipud(loss_mean_8 - loss_std_8)]), color=(0.0,1.0,0.0,0.1))

        N = len(loss_mean_16)
        plt.plot(linspace(0.0, 1.0, N), loss_mean_16, 'r', label="16")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_16 + loss_std_16, flipud(loss_mean_16 - loss_std_16)]), color=(1.0,0.0,0.0,0.1))

        N = len(loss_mean_32)
        plt.plot(linspace(0.0, 1.0, N), loss_mean_32, 'k', label="32")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_32 + loss_std_32, flipud(loss_mean_32 - loss_std_32)]), color=(0.0,0.0,0.0,0.1))

        N = len(loss_mean_64)
        plt.plot(linspace(0.0, 1.0, N), loss_mean_64, label="64", color=(0.0,1.0,1.0,1.0))
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_64 + loss_std_64, flipud(loss_mean_64 - loss_std_64)]), color=(0.0,1.0,1.0,0.1))

        N = len(loss_mean_128)
        plt.plot(linspace(0.0, 1.0, N), loss_mean_128, label="128", color=(1.0,0.0,1.0,1.0))
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([loss_mean_128 + loss_std_128, flipud(loss_mean_128 - loss_std_128)]), color=(1.0,0.0,1.0,0.1))

        plt.xlabel("Ratio L1 / L2")
        plt.ylabel("LOSS (LEARNING)")
        plt.tight_layout()
        plt.legend()
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.ylim([-0.1,1.1])

        # -- ACCURACY
        plt.subplot(132)

        N = len(acc_mean_4)
        plt.plot(linspace(0.0, 1.0, N), acc_mean_4, 'b', label="4")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_4 + acc_std_4, flipud(acc_mean_4 - acc_std_4)]), color=(0,0,1.0,0.1))

        N = len(acc_mean_8)
        plt.plot(linspace(0.0, 1.0, N), acc_mean_8, 'g', label="8")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_8 + acc_std_8, flipud(acc_mean_8 - acc_std_8)]), color=(0.0,1.0,0.0,0.1))

        N = len(acc_mean_16)
        plt.plot(linspace(0.0, 1.0, N), acc_mean_16, 'r', label="16")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_16 + acc_std_16, flipud(acc_mean_16 - acc_std_16)]), color=(1.0,0.0,0.0,0.1))

        N = len(acc_mean_32)
        plt.plot(linspace(0.0, 1.0, N), acc_mean_32, 'k', label="32")
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_32 + acc_std_32, flipud(acc_mean_32 - acc_std_32)]), color=(0.0,0.0,0.0,0.1))

        N = len(acc_mean_64)
        plt.plot(linspace(0.0, 1.0, N), acc_mean_64, label="64", color=(0.0,1.0,1.0,1.0))
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_64 + acc_std_64, flipud(acc_mean_64 - acc_std_64)]), color=(0.0,1.0,1.0,0.1))

        N = len(acc_mean_128)
        plt.plot(linspace(0.0, 1.0, N), acc_mean_128, label="128", color=(1.0,0.0,1.0,1.0))
        x = linspace(0.0, 1.0, N); x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([acc_mean_128 + acc_std_128, flipud(acc_mean_128 - acc_std_128)]), color=(1.0,0.0,1.0,0.1))

        plt.xlabel("Ratio L1 / L2")
        plt.ylabel("ACCURACY (TEST)")
        plt.tight_layout()
        plt.legend()
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.ylim([50.0,100.0])

        # -- COMPLEXITY
        plt.subplot(133)

        N = len(acc_mean_4)
        plt.plot(linspace(0.0, 1.0, N), np4, 'b', label="4")

        N = len(acc_mean_8)
        plt.plot(linspace(0.0, 1.0, N), np8, 'g', label="8")

        N = len(acc_mean_16)
        plt.plot(linspace(0.0, 1.0, N), np16, 'r', label="16")

        N = len(acc_mean_32)
        plt.plot(linspace(0.0, 1.0, N), np32, 'k', label="32")

        N = len(acc_mean_64)
        plt.plot(linspace(0.0, 1.0, N), np64, label="64", color=(0.0,1.0,1.0,1.0))

        N = len(acc_mean_128)
        plt.plot(linspace(0.0, 1.0, N), np128, label="128", color=(1.0,0.0,1.0,1.0))

        plt.xlabel("Ratio L1 / L2")
        plt.ylabel("NUMBER OF PARAMETERS")
        plt.tight_layout()
        plt.legend()
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)

        plt.show()


    if LOAD:
        # -- load data
        X, Y, sigmas = pickle.load(open('data.pkl','rb'))

        # -- normalize data
        newX = data_normalize(X)
        pickle.dump((newX, Y, sigmas), open('data-n.pkl','wb'))
        # print(newX, X)

        # -- pick random samples from the data
        samples, idx = pick_nsamples(X, 10)
        samples0 = samples[Y[idx] == 0, :]
        samples1 = samples[Y[idx] == 1, :]

        # idx, cidx = split_dataset(X)

        # -- display data
        plt.figure(figsize=(12,6))

        # display samples from class 0
        plt.subplot(211)
        plt.title("Class 0 samples")
        plt.plot(samples0.T, lw=1)

        # display samples from class 1
        plt.subplot(212)
        plt.title("Class 1 samples")
        plt.plot(samples1.T, lw=1)

        plt.tight_layout()
        plt.show()

    if DISPLAY_PERFS_DATABASE:
        l_mean_1, l_std_1, l_min_1, l_max_1, a_mean_1, a_std_1, a_min_1, a_max_1 = csv_stat_cv_1model("./results_fc/NORMALISATION/fc_2_128_NO_NORM.csv")
        l_mean_2, l_std_2, l_min_2, l_max_2, a_mean_2, a_std_2, a_min_2, a_max_2 = csv_stat_cv_1model("./results_fc/NORMALISATION/fc_2_128_NORM_1.csv")
        l_mean_3, l_std_3, l_min_3, l_max_3, a_mean_3, a_std_3, a_min_3, a_max_3 = csv_stat_cv_1model("./results_fc/NORMALISATION/fc_2_128_NORM_2.csv")
        l_mean_4, l_std_4, l_min_4, l_max_4, a_mean_4, a_std_4, a_min_4, a_max_4 = csv_stat_cv_1model("./results_fc/NORMALISATION/fc_2_128_SHUFFLE.csv")

        plt.figure(figsize=(12,5))

        # -- LOSS
        plt.subplot(121)

        N = len(l_mean_4); x = linspace(0.0, N, N)
        plt.plot(x, l_mean_4, label="SHUFFLE", color=(0.0,0.0,0.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([l_mean_4 + l_std_4, flipud(l_mean_4 - l_std_4)]), color=(0.0,0.0,0.0,0.1))

        N = len(l_mean_3); x = linspace(0.0, N, N)
        plt.plot(x, l_mean_3, label="NORM2", color=(0.0,0.0,1.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([l_mean_3 + l_std_3, flipud(l_mean_3 - l_std_3)]), color=(0.0,0.0,1.0,0.1))

        N = len(l_mean_2); x = linspace(0.0, N, N)
        plt.plot(x, l_mean_2, label="NORM 1", color=(1.0,0.0,0.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([l_mean_2 + l_std_2, flipud(l_mean_2 - l_std_2)]), color=(1.0,0.0,0.0,0.025))

        N = len(l_mean_1); x = linspace(0.0, N, N)
        plt.plot(x, l_mean_1, label="NO NORM", color=(1.0,0.0,1.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([l_mean_1 + l_std_1, flipud(l_mean_1 - l_std_1)]), color=(1.0,0.0,1.0,0.1))

        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.legend()
        plt.ylabel("LOSS")
        plt.xlabel("EPOCHS")
        # plt.ylim([-0.1, 1.1])

        plt.subplot(122)

        N = len(a_mean_4); x = linspace(0.0, N, N)
        plt.plot(x, a_mean_4, label="SHUFFLE", color=(0.0,0.0,0.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([a_mean_4 + a_std_4, flipud(a_mean_4 - a_std_4)]), color=(0.0,0.0,0.0,0.1))

        N = len(a_mean_3); x = linspace(0.0, N, N)
        plt.plot(x, a_mean_3, label="NORM2", color=(0.0,0.0,1.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([a_mean_3 + a_std_3, flipud(a_mean_3 - a_std_3)]), color=(0.0,0.0,1.0,0.1))

        N = len(a_mean_2); x = linspace(0.0, N, N)
        plt.plot(x, a_mean_2, label="NORM 1", color=(1.0,0.0,0.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([a_mean_2 + a_std_2, flipud(a_mean_2 - a_std_2)]), color=(1.0,0.0,0.0,0.025))

        N = len(a_mean_1); x = linspace(0.0, N, N)
        plt.plot(x, a_mean_1, label="NO NORM", color=(1.0,0.0,1.0,1.0))
        x = concatenate([x, flipud(x)])
        plt.fill(x, concatenate([a_mean_1 + a_std_1, flipud(a_mean_1 - a_std_1)]), color=(1.0,0.0,1.0,0.1))

        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.legend()
        plt.ylabel("ACCURACY")
        plt.xlabel("EPOCHS")
        # plt.ylim([50.0,100.0])

        plt.tight_layout()
        plt.show()
