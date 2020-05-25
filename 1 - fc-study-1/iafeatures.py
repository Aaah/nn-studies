import pickle
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from numpy import absolute, hanning, zeros, mean, var, median, quantile, pi, diff, unwrap, angle
from PyEMD import EMD, Visualisation
from iatools import cli_progress_test

if __name__ == "__main__":

    do_hilbert = False
    do_stats = False
    do_emd = True
    
    # -- load data
    X, Y, sigmas = pickle.load(open('data.pkl','rb'))

    # -- analytic signal (hilbert transform)
    if do_hilbert:
        Y = zeros(X.shape)
        Z = zeros(X.shape)
        w = hanning(X.shape[1])
        for i, x in enumerate(X):
            s = hilbert(x * w)
            Y[i] = absolute(s) # magnitude
            Z[i] = unwrap(angle(s)) # phase

            cli_progress_test(i, X.shape[0], prefix="Processing : Hilbert transforms")

            # plt.figure()
            # inst_freq = diff(phase) / 2*pi
            # plt.subplot(211)
            # plt.plot(x * w)
            # plt.plot(Y[i])
            # plt.subplot(212)
            # plt.plot(Z[i])
            # plt.plot(inst_freq, 'r')
            # plt.show()

        print()
        pickle.dump(Y, open("./features/ft_hilbert_magnitude.pkl", 'wb'))
        pickle.dump(Z, open("./features/ft_hilbert_phase.pkl", 'wb'))

    # -- empirical mode decomposition
    if do_emd:
        emd = EMD()
        NIMFS = 5
        Y = zeros((X.shape[0], NIMFS, X.shape[1]))
        for i, x in enumerate(X):

            if i > 5:
                break

            # compute emd up to NIMFS intrinsic mode functions
            emd.emd(x, max_imf=NIMFS)
            imfs, _ = emd.get_imfs_and_residue()
            
            # pad the result with zeros to have the same number of IMFs
            Y[i, 0:imfs.shape[0], :] = imfs

            cli_progress_test(i, X.shape[0], prefix="Processing : Empirical Modes Decomposition")
            # plt.figure(); plt.plot(Y[i].T); plt.show()

            print()
            # pickle.dump(Y, open("./features/ft_emd5.pkl", 'wb'))

            t = range(X.shape[1])
            vis = Visualisation()
            vis.plot_imfs(imfs=imfs, t=t, include_residue=False)
            vis.plot_instant_freq(t, imfs=imfs)
            vis.show()

    # -- statistics
    if do_stats:
        
        NSTATS = 9
        Y = zeros((X.shape[0], NSTATS))
        y = zeros(NSTATS)
        for i, x in enumerate(X):
            y[0] = mean(x)
            y[1] = var(x)
            y[2] = skew(x)
            y[3] = kurtosis(x)
            y[4] = quantile(x, 0.50)
            y[5] = quantile(x, 0.10)
            y[6] = quantile(x, 0.90)
            y[7] = min(x)
            y[8] = max(x)
            Y[i] = y
            cli_progress_test(i, X.shape[0], prefix="Processing : General statistics")
        
        print()
        pickle.dump(Y, open("./features/ft_stats.pkl", 'wb'))
