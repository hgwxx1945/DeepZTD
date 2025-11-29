def getgacos(filename):
    # read npy
    gacos = np.load(filename)
    gacos = gacos * 1000
    return gacos


def pre(title, gacospath):
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
    idx = testset.idx.reshape(testset.sqlen)
    idx = idx.to(device)
    sampled = sampled * idx
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
    idx = idx.reshape(1, testset.sqlen)

    # print(gacos[0].shape)
    for i in range(1):
        plt.figure()
        plt.plot(Sigma[i, 97:114], 'r')
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
        gt = Y[i, :] * idx[i, :]
        gt = gt[gt != 0]
        m = Miu[i, :] * idx[i, :]
        m = m[m != 0]

        vidx=testset.vidx.numpy()
        gtd=testset.gtd
        gtd=gtd*vidx
        etd=testset.etd
        etd=etd*vidx
        vhd=testset.vhd
        vhd=vhd*vidx
        vhd=vhd[vhd!=0]
        vwd=testset.vwd
        vwd=vwd*vidx
        vwd=vwd[vwd!=0]
        vtd=vwd+vhd

        gmztd = m + vhd
        gmztd = gmztd[gmztd != 0]
        gtd = gtd[gtd != 0]
        vtd = vtd[vtd != 0]
        etd = etd[etd != 0]
        rms = np.sqrt(np.mean((gmztd - gtd) ** 2))
        print(title)
        mbe = np.mean(gmztd - gtd)
        print("Miu:" + str(rms))
        print("Miu bias:" + str(mbe))
        # rms = np.sqrt(np.mean((pre - gt) ** 2))
        # print("GMLSTM:" + str(rms))
        # vmf = X[i, :, 0] * idx[i, :]
        # vmf = vmf[vmf != 0]
        rms = np.sqrt(np.mean((vtd - gtd) ** 2))
        mbe = np.mean(vtd - gtd)
        print("VMF:" + str(rms))
        print("VMF bias:" + str(mbe))
        # era = X[i, :, 1] * idx[i, :]
        # era = era[era != 0]
        rms = np.sqrt(np.mean((etd - gtd) ** 2))
        mbe = np.mean(etd - gtd)
        print("ERA:" + str(rms))
        print("ERA bias:" + str(mbe))

        gacos = getgacos(gacospath)
        vhd = testset.vhd
        #gacoszwd = gacos - vhd
        vidx = testset.vidx.numpy()
        ga = gacos * vidx

        # ga=gacos[i].squeeze()*idx[i,:]
        ga = ga[ga != 0]
        rms = np.sqrt(np.mean((ga - gtd[gtd!=0]) ** 2))
        print("GACOS:" + str(rms))
        mbe = np.mean(ga - gtd[gtd!=0])
        print("GACOS bias:" + str(mbe))
        plt.figure()
        #plot the x ray by date

        import datetime
        from matplotlib.dates import DateFormatter, DayLocator, HourLocator, drange
        date1=datetime.datetime(2022,6,25)
        date2=datetime.datetime(2022,7,10)
        delta = datetime.timedelta(hours=1)
        dates = drange(date1, date2, delta)

        #convert X to vidx length according vidx index, if vi==0, then the value is Nan
        v=np.zeros(vidx.shape)
        e=np.zeros(vidx.shape)
        gac=np.zeros(vidx.shape)
        gm=np.zeros(vidx.shape)
        gnss=np.zeros(vidx.shape)
        sig=np.zeros(vidx.shape)
        j=0
        for i in range(vidx.shape[0]):
            if vidx[i]==0:
                v[i]=np.nan
                e[i]=np.nan
                gac[i]=np.nan
                gm[i]=np.nan
                gnss[i]=np.nan
                sig[i]=np.nan
            else:
                v[i]=vtd[j]
                e[i]=etd[j]
                gac[i]=ga[j]
                gm[i]=gmztd[j]
                gnss[i]=gtd[j]
                sig[i]=Sigma[0,j]
                j+=1
        # combine v e gac gm gnss together
        res=np.zeros((6,vidx.shape[0]))
        res[0,:]=v
        res[1,:]=e
        res[2,:]=gac
        res[3,:]=gm
        res[4,:]=gnss
        res[5,:]=sig
        #np.save(title+'.npy',res)
        return res[:,264:287]


        plt.plot(dates[264:287],v[264:287], 'b', label='VMF')
        plt.plot(dates[264:287],e[264:287], 'g', label='ERA')
        plt.plot(dates[264:287],gnss[264:287], 'r', label='GNSS')
        plt.plot(dates[264:287],gac[264:287], 'm', label='GACOS')
        plt.plot(dates[264:287],gm[264:287], 'k', label='GMLSTM')
        ax = plt.gca()
        ax.set_xlim(dates[264], dates[287])
        ax.xaxis.set_major_locator(HourLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%H'))

        plt.legend(loc='upper right')
        plt.show()

        #plot delta ztd
        plt.figure()
        plt.plot(dates[264:287],v[264:287]-gnss[264:287], 'b', label='VMF')
        plt.plot(dates[264:287],e[264:287]-gnss[264:287], 'g', label='ERA')
        plt.plot(dates[264:287],gac[264:287]-gnss[264:287], 'm', label='GACOS')
        plt.plot(dates[264:287],gm[264:287]-gnss[264:287], 'k', label='GMLSTM')
        ax = plt.gca()
        ax.set_xlim(dates[264], dates[287])
        ax.xaxis.set_major_locator(HourLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%H'))
        plt.legend(loc='upper right')
        plt.show()


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

filename='/home/wanduo/pyZTD/Spain.mat'
testsiteidx=3
testset = DatafromMat(filename=filename,testsiteidx=testsiteidx,train=False, season='summer')
model = GMLSTM(2048,5,2)
#load model
model.load_state_dict(torch.load('./GMLSTM_Spain_best_day_summer.pth'))
title='Burgos summer'
gacospath='/home/wanduo/GACOSdownload/Burgossummer.npy'
res1=pre(title,gacospath)
filename='/home/wanduo/pyZTD/zaragoza.mat'
testsiteidx=8
testset = DatafromMat(filename=filename,testsiteidx=testsiteidx,train=False, season='winter')
model = GMLSTM(2048,5,2)
#load model
model.load_state_dict(torch.load('./GMLSTM_zaragoza_best_day_summer.pth'))
title='Sala winter'
gacospath='/home/wanduo/GACOSdownload/Zaragozasummer.npy'
res2=pre(title,gacospath)
ANNpath='./Result/ANN/'
#subplot 2,2
fig, axs = plt.subplots(2, 2, figsize=(16,12))
#burgos summer
res=res1
annres = np.load(ANNpath + 'Burgos' + ' Summer.npy')
v=res[0,:]
e=res[1,:]
gac=res[2,:]
gm=res[3,:]
ann=annres[3,:]
gnss=res[4,:]
sig=res[5,:]
#dates
import datetime
from matplotlib.dates import DateFormatter, DayLocator, HourLocator, drange
date1 = datetime.datetime(2022, 6, 25)
date2 = datetime.datetime(2022, 7, 10)
delta = datetime.timedelta(hours=1)
dates = drange(date1, date2, delta)
dates=dates[264:287]
#plot burgos summer
axs[0,0].plot(dates,v, 'b', label='VMF3')
axs[0,0].plot(dates,e, 'g', label='ERA5')

axs[0,0].plot(dates,gac, 'k', label='GACOS')
axs[0,0].plot(dates,gm, 'r', label='GM-LSTM')
ann=ann[264:287]
axs[0,0].plot(dates,ann, 'y', label='DNN')
axs[0,0].plot(dates,gnss, 'm', label='GNSS',linestyle='dashed')
axs[0,0].set_title('ZTD on Jul 6, 2022, Burgos',fontweight='bold',fontsize=20)
axs[0,0].xaxis.set_major_locator(HourLocator(interval=2))
axs[0,0].xaxis.set_major_formatter(DateFormatter('%H'))
axs[0,0].set_xlim(dates[0], dates[-1])

# y axis
axs[0,0].set_ylabel('ZTD (mm)',fontweight='bold',fontsize=20)
axs[0,0].set_ylim(2190,2290)
# x axis
axs[0,0].set_xlabel('Hour',fontweight='bold',fontsize=20)
# ticks font size 20 and bold
axs[0,0].tick_params(axis='both', which='major', labelsize=20)
for tick in axs[0,0].get_xticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
for tick in axs[0,0].get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
#plot burgos sig
axs[0,1].plot(dates,sig, 'r', label='Sigma')
axs[0,1].set_title('Standard deviation of GM-LSTM\n on Jul 6, 2022, Burgos',fontweight='bold',fontsize=20)
axs[0,1].xaxis.set_major_locator(HourLocator(interval=2))
axs[0,1].xaxis.set_major_formatter(DateFormatter('%H'))
axs[0,1].set_xlim(dates[0], dates[-1])
# y axis

axs[0,1].set_ylabel('s (mm)',fontweight='bold',fontsize=20)
# x axis
axs[0,1].set_xlabel('Hour',fontweight='bold',fontsize=20)

axs[0,1].set_ylim(0,20)
# ticks font size 20
axs[0,1].tick_params(axis='both', which='major', labelsize=20)
for tick in axs[0,1].get_xticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
for tick in axs[0,1].get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
res=res2
annres = np.load(ANNpath + 'Zaragoza' + ' Summer.npy')
v=res[0,:]
e=res[1,:]
gac=res[2,:]
gm=res[3,:]
ann=annres[3,:]
ann=ann[264:287]
gnss=res[4,:]
sig=res[5,:]

axs[1,0].plot(dates,v, 'b', label='VMF3')
axs[1,0].plot(dates,e, 'g', label='ERA')

axs[1,0].plot(dates,gac, 'k', label='GACOS')
axs[1,0].plot(dates,gm, 'r', label='GM-LSTM')
axs[1,0].plot(dates,ann, 'y', label='DNN')
axs[1,0].plot(dates,gnss, 'm', label='GNSS',linestyle='dashed')
axs[1,0].set_title('ZTD on Jul 6, 2022, Zaragoza',fontweight='bold',fontsize=20)
axs[1,0].xaxis.set_major_locator(HourLocator(interval=2))
axs[1,0].xaxis.set_major_formatter(DateFormatter('%H'))
# y axis
axs[1,0].set_ylabel('ZTD (mm)',fontweight='bold',fontsize=20)
# x axis
axs[1,0].set_xlabel('Hour',fontweight='bold',fontsize=20)
axs[1,0].set_xlim(dates[0], dates[-1])
axs[1,0].set_ylim(2420,2540)
# ticks font size 20
axs[1,0].tick_params(axis='both', which='major', labelsize=20)
for tick in axs[1,0].get_xticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
for tick in axs[1,0].get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
#plot burgos sig
axs[1,1].plot(dates,sig, 'r', label='Sigma')
axs[1,1].set_title('Standard deviation of GM-LSTM\n on Jul 6, 2022, Zaragoza',fontweight='bold',fontsize=20)
axs[1,1].xaxis.set_major_locator(HourLocator(interval=2))
axs[1,1].xaxis.set_major_formatter(DateFormatter('%H'))
# y axis
axs[1,1].set_ylabel('s (mm)',fontweight='bold',fontsize=20)
# x axis
axs[1,1].set_xlabel('Hour',fontweight='bold',fontsize=20)

axs[1,1].set_ylim(0,20)
axs[1,1].set_xlim(dates[0], dates[-1])
# ticks font size 20
axs[1,1].tick_params(axis='both', which='major', labelsize=20)
for tick in axs[1,1].get_xticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)
for tick in axs[1,1].get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(20)

#add (a) (b) (c) (d) for subfigure
axs[0,0].text(0, 1, '(a)', transform=axs[0,0].transAxes, fontsize=15, verticalalignment='top')
axs[0,1].text(0, 1, '(b)', transform=axs[0,1].transAxes, fontsize=15, verticalalignment='top')
axs[1,0].text(0, 1, '(c)', transform=axs[1,0].transAxes, fontsize=15, verticalalignment='top')
axs[1,1].text(0, 1, '(d)', transform=axs[1,1].transAxes, fontsize=15, verticalalignment='top')

# add fulfill color between 12:00 to 14:30 of each subfigure
for i in range(2):
    ymin, ymax = axs[0,i].get_ylim()
    axs[0,i].fill_between(dates[12:16],ymin,ymax,facecolor='gray',alpha=0.5)
    ymin, ymax = axs[1,i].get_ylim()
    axs[1,i].fill_between(dates[12:16],ymin,ymax,facecolor='gray',alpha=0.5)

#legend below
handles, labels = axs[0,0].get_legend_handles_labels()
#add that gray squeare label
handles.append(plt.Rectangle((0,0),1,1,fc="gray",alpha=0.5))
labels.append('Time affected by rainfall')
#bold the legend
fig.legend(handles, labels,ncol=7, loc='lower center',prop=dict(weight='bold',size='15'))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.4,wspace=0.2)
plt.savefig('Fig9.jpg',dpi=600)
plt.show()
#save high rsolution figure
