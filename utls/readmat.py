#load data from matlab file
import h5py
import numpy
import torch
from torch.utils.data import Dataset, DataLoader

def readmat(filename):
    with h5py.File(filename, 'r') as f:
        trainx = f['trainx'][:]
        trainy = f['trainy'][:]
        testx = f['xtest'][:]
        testy = f['ytest'][:]
        valx= f['valx'][:]
        valy = f['valy'][:]
    return trainx, trainy, testx, testy,valx,valy
def readgacos():
    with h5py.File('./D400GACOSZTD.mat', 'r') as f:
        gacosztd = f['GACOSZTD'][:]
        era5hd = f['era5400_HD'][:]
    return gacosztd,era5hd
#main
if __name__ == '__main__':
    trainx, trainy, testx, testy= readmat('./Karldata.mat')
    print(trainx.shape)
    print(trainy.shape)
    print(testx.shape)
    print(testy.shape)