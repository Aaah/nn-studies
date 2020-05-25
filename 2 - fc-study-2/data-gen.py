import numpy as np
import pylab as pl
import random as rdm
import pickle
from iatools import shuffle_dataset
from scipy.spatial import distance

def two_pounds_fn(centers, radius, in_bool, npoints):
    """
    Create dataset with 2 classes, 1 being split between two pounds and the other being the rest of the space.
    """
    arr = np.zeros((npoints, 2))

    n = 0
    while n < npoints:
        p = np.random.uniform(-6.0, 6.0, size=(2,))
        d1 = distance.euclidean( p, centers[0] )
        d2 = distance.euclidean( p, centers[1] )
        if d1 <= radius[0] or d2 <= radius[1]:
            if in_bool == True:
                arr[n] = p
                n += 1
        else:
            if in_bool == False:
                arr[n] = p
                n += 1
    
    return arr


def spiral_fn(center, radius, phase, turns, std, n):
    """
    center : position of the centre of the spiral
    radius : extent of the spiral from the center
    phase : phase at origin
    turns : number of turns from the center to the end
    std : standard deviation
    n : number of points
    """

    alpha = 2.0 * np.pi * turns
    beta = 1.0 * radius

    arr = np.zeros((npoints, 2))
    for n in range(npoints):
        t = np.random.uniform(low=0.0, high=1.0)
        arr[n] = [beta * t * np.cos(alpha * t + phase), beta * t * np.sin(alpha * t + phase)]
        arr[n] = arr[n] + np.random.normal(0, std, 2)

    return arr

def dot_fn(x, y, stdx, stdy, n):

    arr = np.zeros((npoints, 2))
    for n in range(npoints):
        arr[n] = [np.random.normal(x, stdx, 1), np.random.normal(y, stdy, 1)]

    return arr

def u_fn(x, y, radius, angle, stdx, stdy, n):
    
    arr = np.zeros((npoints, 2))
    for n in range(npoints):

        u = np.random.uniform(angle, 1.0 - angle)
        x = radius * np.cos(2 * np.pi * u)
        y = radius * np.sin(2 * np.pi * u)
        arr[n] = [np.random.normal(x, stdx, 1), np.random.normal(y, stdy, 1)]

    return arr

if __name__ == "__main__":  

    DISPLAY = True
    SAVE = True
    example_name = "pounds"

    # x = 0.0
    # y = 0.0
    # stdx = 0.2
    # stdy = 0.2
    # npoints = 1000
    # angle = 0.0
    # radius = 4.0
    centers = [[-2.0, 0.0],[+2.0,0.0]]
    radius = [1.0, 0.5]
    npoints = 1000
    data1 = two_pounds_fn(centers, radius, True, 1000)
    data2 = two_pounds_fn(centers, radius, False, 1000)

    # -- agregate data 1
    # data1 = dot_fn(x, y, 2 * stdx, 2 * stdy, npoints)
    # data2 = u_fn(x, y, radius, angle, stdx, stdy, npoints)

    # data1 = two_pounds_fn(center=(0,0), radius=5.0, phase=0.0, turns=2.0, std=0.1, n=npoints)
    # data2 = spiral_fn(center=(0,0), radius=5.0, phase=np.pi, turns=2.0, std=0.1, n=npoints)

    data = np.concatenate([data1, data2])
    y = np.concatenate([np.zeros(npoints), np.ones(npoints)])

    # -- shuffle data
    idx = shuffle_dataset(y)
    data = data[idx]
    y = y[idx]

    if SAVE:
        pkl_file_name = "./2d-data-"+example_name+".pkl"
        pickle.dump([data, y], open(pkl_file_name, "wb"))
        _data, _y = pickle.load(open(pkl_file_name,'rb'))

    if DISPLAY:
        pl.figure(figsize=(4,4))
        pl.plot(data1.T[0], data1.T[1], '.', color="gray", markersize=1.0, label="class 0")
        pl.plot(data2.T[0], data2.T[1], '.', color="red", markersize=1.0, label="class 1")
        pl.xlim([-5.0, +5.0])
        pl.ylim([-5.0, +5.0])
        pl.grid()
        pl.legend()
        pl.tight_layout()
        pl.ylim([-6.0, 6.0])
        pl.xlim([-6.0, 6.0])
        pl.savefig("./2d-data-" + example_name + ".pdf")
        pl.show()

