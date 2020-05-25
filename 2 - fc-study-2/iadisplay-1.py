from iatools import csv_get_every_instance
import matplotlib.pyplot as plt
import sys
import csv
from numpy import array, linspace, zeros, mean, std, min, concatenate, argsort, flipud, argmax, arange

if __name__ == '__main__':
    
    # models = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    models = {20}

    for m in models:
        learn_d, test_d = csv_get_every_instance("./TEMP/fcnn_1_" + str(m) + "_DUMP.csv")

        plt.figure(figsize=(10,5))

        plt.suptitle("CONVERGENCE WITH NHIDDEN = " + str(m))

        # -- LOSS
        plt.subplot(121)
        plt.plot(learn_d.T)
        plt.xlabel("EPOCHS")
        plt.ylabel("LOSS (LEARNING)")
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.ylim([0.0,1.0])

        # -- ACCURACY
        plt.subplot(122)
        plt.plot(test_d.T)
        plt.xlabel("EPOCHS")
        plt.ylabel("ACCURACY (TEST)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.grid(color=(0.75,0.75,0.75), linestyle='--', linewidth=1)
        plt.ylim([40.0,100.0])

        plt.savefig("./results/convergence-"+ str(m) +".pdf")
        plt.show()