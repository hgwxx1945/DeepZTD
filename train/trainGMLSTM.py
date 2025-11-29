import  torch
import torch.nn.functional as F

from train1dUnet import valset
from utls.ztddataloader import DatafromMat
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.GMLSTM import GMLSTM,gmlstm_loss_function_rnn

season='winter'
filename='/home/wanduo/pyZTD/zaragoza.mat'
testsiteidx=8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def prepare_data(season):
    seed=np.random
    trainset = DatafromMat(filename=filename,testsiteidx=testsiteidx,train=True,season=season,val=False,val_split=0.1,seed=seed)
    valset = DatafromMat(filename=filename,testsiteidx=testsiteidx,train=False, season=season,val=True,val_split=0.1,seed=seed)
    return trainset,valset
def initraining(trainset,valset):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=300, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=15, shuffle=False)
    model = GMLSTM(2048,5,2)

    model.to(device)

    q = torch.tensor([0.2, 0.4, 0.6, 0.8, 1])

    model.z_mu.bias.data = torch.quantile(trainset.Y, q).to(device)
    model.z_mu.bias.requires_grad = True
    lr=0.001


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    return train_loader,val_loader,model,optimizer,scheduler

def train(epoch):
    model.train()
    trainloss=0
    for batch_idx, data in enumerate(train_loader):
        X,Y=data
        X=X.to(device)
        Y=Y.to(device)
        optimizer.zero_grad()
        # #GMLSTM loss
        (out_pi, out_sigma, out_mu) = model(X)
        loss = gmlstm_loss_function_rnn(out_pi, out_sigma, out_mu, Y)
        Miu = torch.sum(out_mu * out_pi, 2)
        mseloss = F.mse_loss(Miu, Y.squeeze())
        # mseloss=F.mse_loss(out_mu,Y)
        # loss+=mseloss
        loss.backward()
        optimizer.step()
        if epoch%500==0:
            scheduler.step()

        trainloss+=loss.item()




    #polt predict for every 100 epoch
    if epoch % 1000 == 0:
        # plt.cla()
        # plt.plot(Y[0, :].detach().squeeze().to(torch.device('cpu')).numpy(), 'bs')
        # plt.plot(model(X)[0, :].detach().squeeze().to(torch.device('cpu')).numpy(), 'g-')
        #
        # plt.show()
        # plt.pause(0.01)
        print('====> Epoch: {} train loss: {:.4f}'.format(
            epoch, mseloss.item() ))


def val(epoch,bestloss,season,bestepoch):
    if season=='winter':
        savename='./GMLSTM_zaragoza_best_day_summer.pth'
    else:
        savename='./GMLSTM_zaragoza_best_day_summer.pth'
    valloss=0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            X, Y = data
            X = X.to(device)
            Y = Y.to(device)

            # predict
            model.eval()
            model.to(device)
            pi_variable, sigma_variable, mu_variable = model(X)
            pi = pi_variable.data.squeeze()
            sigma = sigma_variable.data.squeeze()
            mu = mu_variable.data.squeeze()
            # select the most likely gaussian
            pi = pi.reshape(-1, 5)
            sigma = sigma.reshape(-1, 5)
            mu = mu.reshape(-1, 5)

            # idx = testset.idx.reshape(testset.sqlen)
            # sampled = sampled * idx

            Miu = torch.sum(mu * pi, axis=1)
            Sigma2 = torch.sum(((mu - Miu) ** 2 + sigma ** 2) * pi, axis=1)# adjust reshape Miu.reshape(valset.sqlen, 1) as needed
            Sigma = torch.sqrt(Sigma2)
            Ynonzero = Y[Y != 0]
            loss = torch.sqrt(F.mse_loss(Miu, Y.reshape(-1).squeeze())) + torch.mean(Sigma)
            valloss += loss.item()
        if valloss < bestloss:
            bestepoch = epoch
            bestloss = valloss
            torch.save(model.state_dict(), savename)
            print('====> Epoch: {} best loss: {:.4f} mean std {:.4f}'.format(
                epoch, valloss, torch.mean(Sigma).item()))

    #polt predict for every 100 epoch
    if epoch%200==0:
        print('====> Epoch: {} test loss: {:.4f} mean std {:.4f}'.format(
            epoch, valloss, torch.mean(Sigma).item()))
    return bestloss,bestepoch


season='winter'
trainset,valset=prepare_data(season)
train_loader,val_loader,model,optimizer,scheduler=initraining(trainset,valset)
bestloss=100000
bestepoch=0
for epoch in range(1, 10000):
    train(epoch)
    bestloss,bestepoch=val(epoch,bestloss=bestloss,season=season,bestepoch=bestepoch)
# save best epoch into txt
with open('bestepoch_zaragoza.txt', 'a') as f:
    f.write('best epoch for zaragoza is '+str(bestepoch)+'\n')

torch.save(model.state_dict(), './GMLSTM_zaragoza_day_summer.pth')
# season='summer'
# trainset,testset=prepare_data(season)
# train_loader,test_loader,model,optimizer,scheduler=initraining(trainset,testset)
# bestloss=100000
# bestepoch=0
# for epoch in range(1, 10000):
#     train(epoch)
#     bestloss,bestepoch=test(epoch,bestloss=bestloss,season=season,bestepoch=bestepoch)
# torch.save(model.state_dict(), './GMLSTM_svap_day_summer.pth')
# # save best epoch into txt
# with open('bestepoch_svap.txt', 'a') as f:
#     f.write('best epoch for summer Svap is '+str(bestepoch)+'\n')

