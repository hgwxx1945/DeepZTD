from utls.ztddataloader import DatafromMat
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.GMLSTM import GMLSTM
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def gumbel_sample(x, axis=1):
    z=torch.distributions.gumbel.Gumbel(0,1).sample(x.shape).to(device)

    #z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return torch.argmax(torch.log(x) + z,dim=axis)

def getgacos(filename):
    # read npy
    gacos = np.load(filename)
    gacos = gacos * 1000
    return gacos


def pre(title, gacospath):
    matrics = []
    # for whole sq
    X = testset.X
    Y = testset.Y
    # X=X.reshape(2,360,2)
    # Y=Y.reshape(2,360)
    idx = testset.idx
    # predict
    model.eval()
    model.to(device)
    X = torch.tensor(X).float().to(device)
    Y = torch.tensor(Y).float().to(device)
    pi_variable, sigma_variable, mu_variable = model(X)
    pi = pi_variable.data.squeeze()
    sigma = sigma_variable.data.squeeze()
    mu = mu_variable.data.squeeze()
    # select the most likely gaussian
    pi = pi.reshape(-1, 5)
    sigma = sigma.reshape(-1, 5)
    mu = mu.reshape(-1, 5)
    k = gumbel_sample(pi)
    size = pi.shape[0]
    indices = (torch.arange(size), k)
    rn = torch.randn(size).to(device)
    sampled = rn * sigma[indices] + mu[indices]
    # idx = testset.idx.reshape(testset.sqlen)
    idx = idx.to(device)
    # sampled = sampled * idx
    samplednonzero = sampled[sampled != 0]
    mu_data = mu.cpu().numpy()
    sigma_data = sigma.cpu().numpy()
    pi_data = pi.cpu().numpy()
    Miu = np.sum(mu_data * pi_data, axis=1)
    Sigma2 = np.sum(((mu_data - Miu.reshape(testset.sqlen, 1)) ** 2 + sigma_data ** 2) * pi_data, axis=1)
    Sigma = np.sqrt(Sigma2)

    halfsigma = Sigma.squeeze() / 2
    lower = Miu.squeeze() - halfsigma
    upper = Miu.squeeze() + halfsigma
    Miu = Miu.reshape(1, testset.sqlen)
    Sigma = Sigma.reshape(1, testset.sqlen)
    lower = lower.reshape(1, testset.sqlen)
    upper = upper.reshape(1, testset.sqlen)
    # plot
    sampled = sampled.cpu().numpy()
    sampled = sampled.reshape(1, testset.sqlen)
    X = X.reshape(1, testset.sqlen, 2)
    Y = Y.reshape(1, testset.sqlen)
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    idx = idx.cpu().numpy()
    # idx=idx[idx!=0]
    # idx = idx.reshape(1, testset.sqlen)

    # print(gacos[0].shape)
    for i in range(1):
        plt.figure()
        plt.plot(Sigma[i, :], 'r')
        # plt.plot(Miu[i,:]-Y[i,:],'r')
        plt.figure()

        # plt.plot(X[i,:,0]*idx[i,:],'b')
        # plt.plot(X[i,:,1]*idx[i,:],'ro')
        # plt.plot(Y[i,:]*idx[i,:],'g')
        plt.plot(sampled[i, :], 'k')
        xaxis = np.arange(0, upper.shape[1])
        plt.fill_between(xaxis, lower[i, :], upper[i, :], color='red', alpha=1)
        plt.show()
        # calc rms remove 0
        pre = sampled[i, :]
        pre = pre[pre != 0]
        # gt = Y[i, :] * idx
        # gt = gt[gt != 0]
        m = Miu[i, :] * idx
        m = m[m != 0]

        vidx = testset.vidx.numpy()
        gtd = testset.gtd
        gtd = gtd * vidx
        etd = testset.etd
        etd = etd * vidx
        vhd = testset.vhd
        vhd = vhd * vidx
        vhd = vhd[vhd != 0]
        vwd = testset.vwd
        vwd = vwd * vidx
        vwd = vwd[vwd != 0]
        vtd = vwd + vhd

        gmztd = m + vhd
        gtd = gtd[gtd != 0]
        vtd = vtd[vtd != 0]
        etd = etd[etd != 0]
        rms = np.sqrt(np.mean((gmztd - gtd) ** 2))
        matrics.append(rms)
        print(title)
        mbe = np.mean(gmztd - gtd)
        matrics.append(mbe)
        print("Miu:" + str(rms))
        print("Miu bias:" + str(mbe))
        std = np.std(gmztd - gtd)
        matrics.append(std)
        print("Miu std:" + str(std))
        # rms = np.sqrt(np.mean((pre - gt) ** 2))
        # print("GMLSTM:" + str(rms))
        # vmf = X[i, :, 0] * idx[i, :]
        # vmf = vmf[vmf != 0]
        rms = np.sqrt(np.mean((etd - gtd) ** 2))
        matrics.append(rms)
        mbe = np.mean(etd - gtd)
        matrics.append(mbe)
        std=np.std(etd - gtd)
        matrics.append(std)
        print("ERA:" + str(rms))
        print("ERA bias:" + str(mbe))
        print("ERA std:" + str(std))
        rms = np.sqrt(np.mean((vtd - gtd) ** 2))
        matrics.append(rms)
        mbe = np.mean(vtd - gtd)
        matrics.append(mbe)
        std = np.std(vtd - gtd)
        matrics.append(std)
        print("VMF:" + str(rms))
        print("VMF bias:" + str(mbe))
        print("VMF std:" + str(std))
        # era = X[i, :, 1] * idx[i, :]
        # era = era[era != 0]


        gacos = getgacos(gacospath)

        # gacoszwd=gacos-vhd

        ga = gacos * vidx

        # ga=gacos[i].squeeze()*idx[i,:]
        ga = ga[ga != 0]
        rms = np.sqrt(np.mean((ga - gtd) ** 2))
        matrics.append(rms)
        print("GACOS:" + str(rms))
        mbe = np.mean(ga - gtd)
        matrics.append(mbe)
        print("GACOS bias:" + str(mbe))
        std = np.std(ga - gtd)
        matrics.append(std)
        print("GACOS std:" + str(std))
        v = np.zeros(vidx.shape)
        e = np.zeros(vidx.shape)
        gac = np.zeros(vidx.shape)
        gm = np.zeros(vidx.shape)
        gnss = np.zeros(vidx.shape)
        j = 0
        for i in range(vidx.shape[0]):
            if vidx[i] == 0:
                v[i] = np.nan
                e[i] = np.nan
                gac[i] = np.nan
                gm[i] = np.nan
                gnss[i] = np.nan
            else:
                v[i] = vtd[j]
                e[i] = etd[j]
                gac[i] = ga[j]
                gm[i] = gmztd[j]
                gnss[i] = gtd[j]
                j += 1
        # save file based on title name
        res = np.zeros((5, vidx.shape[0]))
        res[0, :] = v
        res[1, :] = e
        res[2, :] = gac
        res[3, :] = gm
        res[4, :] = gnss
        np.save(title+'.npy',res)

        plt.figure()
        plt.plot(v, 'b', label='VMF')
        plt.plot(e, 'g', label='ERA')
        plt.plot(gnss, 'r', label='GNSS')
        plt.plot(gac, 'm', label='GACOS')
        plt.plot(gm, 'k', label='GMLSTM')

        plt.legend(loc='upper right')
        return res,matrics

filepath='/home/wanduo/pyZTD/'
filelist=['Portugal.mat','Spain.mat','Paris.mat','France.mat','Tuebingen.mat','Niderland.mat','sala.mat','sweden.mat']
modellist=['./GMLSTM_Portugal_best_day_', './GMLSTM_Spain_best_day_', './GMLSTM_Paris_best_day_', './GMLSTM_francebriv_best_day_', './GMLSTM_Tuebingen_best_day_', './GMLSTM_Nider_best_day_', './GMLSTM_sala_best_day_', './GMLSTM_svap_best_day_']
titlelist=['Beja','Burgos','Paris','Brive','Tuebingen','Groningen','Sala','Svap']
testid=[3,3,7,0,5,3,3,7]
import pandas as pd
#creat datafram with title GM-LSTMRMS GM_LSTMMBE GM_LSTMSTD ERA5RMS ERA5MBE ERA5STD VMFRMS VMFMBE VMFSTD GACOSRMS GACOSMBE GACOSSTD
df=pd.DataFrame(columns=['GM-LSTMRMS','GM_LSTMMBE','GM_LSTMSTD','ERA5RMS','ERA5MBE','ERA5STD','VMFRMS','VMFMBE','VMFSTD','GACOSRMS','GACOSMBE','GACOSSTD'])


for i in range(8):
    filename=filepath+filelist[i]
    testsiteidx=testid[i]

    if i is not 2:
        wintermodelpath=modellist[i]+'winter.pth'
        summermodelpath=modellist[i]+'summer.pth'
        winter='winter'
        summer='summer'
    else:
        wintermodelpath=modellist[i]+'winter.pth'
        summermodelpath=modellist[i]+'summer.pth'
        winter='summer'
        summer='winter'
    #load model
    testset = DatafromMat(filename=filename, testsiteidx=testsiteidx, train=False, season=winter)
    model = GMLSTM(2048, 5, 2)
    model.load_state_dict(torch.load(wintermodelpath)) #_zone_gbsp15
    title=titlelist[i]+' Winter'
    gacospath='/home/wanduo/GACOSdownload/'+titlelist[i]+'winter.npy'
    res,matrics=pre(title,gacospath)
    df.loc[2*i]=matrics

    testset = DatafromMat(filename=filename,testsiteidx=testsiteidx,train=False, season=summer)
    model = GMLSTM(2048,5,2)
    #load model
    model.load_state_dict(torch.load(summermodelpath))
    title=titlelist[i]+' Summer'
    gacospath='/home/wanduo/GACOSdownload/'+titlelist[i]+'summer.npy'
    res,matrics=pre(title,gacospath)
    df.loc[2*i+1]=matrics

print(df)
#save df as csv
df.to_csv('test8area.csv')