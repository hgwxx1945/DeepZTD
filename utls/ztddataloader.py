#create a data loader from mat file

from utls.readmat import readmat
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
import torch


class DatafromMat_LLH(Dataset):
    def __init__(self,filename,testsiteidx, train=True,season='winter'):
        self.train = train
        import h5py
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
            etd = f['etd'][:]
            llh = f['llh'][:]
        if season == 'winter':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
            etd = etd[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
            etd = etd[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        etd = etd * day


        vwd = vwd * day

        ewd = ewd * day

        llh = np.expand_dims(llh.T, 2)
        llh = np.repeat(llh, 360, axis=2)
        llh = llh*day

        #idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 6*360
            vin = np.expand_dims(vin, 0)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = np.expand_dims(ein, 0)
            llh = np.delete(llh, testsiteidx, 1)

            #concatenate vin and ein and llh
            xin = np.concatenate((vin, ein), 0)
            xin = np.concatenate((xin, llh), 0)
            # 1*7*360+1*7*360+3*7*360 -> 5*7*360 then remove the row of vin==0
            xin = xin.reshape(5, -1)
            #delete zero column
            xin = xin[:, ~np.all(xin == 0, axis=0)]


            self.X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[1]
        else:
            vin = vwd[testsiteidx, :]
            vin = np.expand_dims(vin, 0)

            ein = ewd[testsiteidx, :]
            ein = np.expand_dims(ein, 0)
            llh = llh[:,testsiteidx, :]


            self.sqlen = int(sum(day[testsiteidx, :]))
            xin = np.concatenate((vin, ein), 0)
            xin = np.concatenate((xin, llh), 0)

            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]

            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[1]
            self.vidx=idx[testsiteidx, :]*day[testsiteidx, :]
            self.vidx=torch.tensor(self.vidx, dtype=torch.float32)
            self.idx = idx[testsiteidx, :]
            self.idx = torch.tensor(self.idx, dtype=torch.float32)
            vhd=vhd[testsiteidx,:]
            self.vhd=vhd
            vwd=vwd[testsiteidx,:]
            self.vwd=vwd
            etd=etd[testsiteidx,:]
            self.etd=etd
            gtd=gtd[testsiteidx,:]
            self.gtd=gtd

        self.Y = self.Y.unsqueeze(1)

    def __getitem__(self, index):
        if self.train:
            X = self.X[:,index]
            Y = self.Y[index,:]
        else:
            X = self.X[:,index]
            Y = self.Y[index,:]
        return X, Y
    def __len__(self):
        return self.len

class DatafromMat(Dataset):
    def __init__(self,filename,testsiteidx, train=True, season='winter', val=False, val_split=0.1, seed=None):
        self.train = train
        self._use_val = bool(val)  # if True and train==True, use the validation subset
        self._val_split = float(val_split)
        self._seed = seed
        import h5py
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
            etd = f['etd'][:]
        if season == 'winter':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
            etd = etd[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
            etd = etd[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        etd = etd * day


        vwd = vwd * day

        ewd = ewd * day

        #idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 7*360
            vin = vin[vin != 0]
            # reshape N by 24 hours
            vin = vin.reshape(-1, 24)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = ein[ein != 0]
            ein = ein.reshape(-1, 24)
            # stack
            xin = np.stack((vin, ein), 2)  # N*24*2
            # 360-> every 24 hours

            full_X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            full_Y = torch.tensor(gout, dtype=torch.float32)

            # if validation splitting is requested, split into train / val
            if self._val_split is not None and self._val_split > 0:
                N = full_X.shape[0]
                # determine number of validation samples (at least 1 if possible)
                val_count = int(round(self._val_split * N))
                if val_count < 1 and N > 1:
                    val_count = 1
                if val_count >= N:
                    val_count = max(1, N // 10)

                # create shuffled indices reproducibly when seed provided
                rng = np.random.RandomState(self._seed) if self._seed is not None else np.random
                indices = np.arange(N)
                rng.shuffle(indices)
                val_idx = indices[:val_count]
                train_idx = indices[val_count:]

                if self._use_val:
                    # expose validation subset
                    self.X = full_X[val_idx]
                    self.Y = full_Y[val_idx]
                else:
                    # expose training subset
                    self.X = full_X[train_idx]
                    self.Y = full_Y[train_idx]
            else:
                # no split requested, use full set as before
                self.X = full_X
                self.Y = full_Y

            self.len = self.X.shape[0]
        else:
            vin = vwd[testsiteidx, :]
            vin = vin[vin != 0]
            ein = ewd[testsiteidx, :]
            ein = ein[ein != 0]
            self.sqlen = int(sum(day[testsiteidx, :]))
            vin = vin.reshape(-1, 24)
            ein = ein.reshape(-1, 24)
            xin = np.stack((vin, ein), 2)
            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.vidx=idx[testsiteidx, :]*day[testsiteidx, :]
            self.vidx=torch.tensor(self.vidx, dtype=torch.float32)
            self.idx = idx[testsiteidx, :].reshape(-1,24)
            d24=day[testsiteidx, :].reshape(-1,24)
            zeroline=np.all(d24==0,1)
            #delete the vector of whole 0
            self.idx=self.idx[~zeroline]
            #flat
            self.idx=self.idx.reshape(-1)
            self.idx = torch.tensor(self.idx, dtype=torch.float32)
            vhd=vhd[testsiteidx,:]
            self.vhd=vhd
            vwd=vwd[testsiteidx,:]
            self.vwd=vwd
            etd=etd[testsiteidx,:]
            self.etd=etd
            gtd=gtd[testsiteidx,:]
            self.gtd=gtd

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index, :]
            Y = self.Y[index, :]
        else:
            X = self.X[index, :]
            Y = self.Y[index, :]
        return X, Y

    def __len__(self):
        return self.len


class Karl_old(Dataset):
    def __init__(self, train=True,val=False):
        trainx, trainy, testx, testy,valx,valy= readmat('./Karldata_new.mat')
        if train:
            self.X = torch.tensor(trainx, dtype=torch.float32)
            self.Y = torch.tensor(trainy, dtype=torch.float32)
        else:
            self.X = torch.tensor(testx, dtype=torch.float32)
            self.Y = torch.tensor(testy, dtype=torch.float32)
        if val:
            self.X = torch.tensor(valx, dtype=torch.float32)
            self.Y = torch.tensor(valy, dtype=torch.float32)
        self.len = self.X.shape[0]
        #unsqueeze
        self.X = self.X.unsqueeze(1)
        self.Y = self.Y.unsqueeze(1)

    def __getitem__(self, index):
            X = self.X[index,:]
            Y = self.Y[index,:]
            return X, Y

    def __len__(self):
        return self.len
class Sweden(Dataset):
    def __init__(self, train=True,season='winter'):
        self.train = train
        import h5py
        filename = './sweden.mat'
        with h5py.File(filename, 'r') as f:
            gtd= f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
        if season=='winter':
            gtd=gtd[:,0:360]
            gwd=gwd[:,0:360]
            vhd=vhd[:,0:360]
            vwd=vwd[:,0:360]
            ewd=ewd[:,0:360]
            day=idxday[:,0:360]
            idx=idx[:,0:360]
        else:
            gtd=gtd[:,360:720]
            gwd=gwd[:,360:720]
            vhd=vhd[:,360:720]
            vwd=vwd[:,360:720]
            ewd=ewd[:,360:720]
            day=idxday[:,360:720]
            idx=idx[:,360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        vwd = vwd * day

        ewd = ewd * day

        idx=idx*day
        if train:
            vin=np.delete(vwd,[7],0) #7*360
            vin=vin[vin!=0]
            #reshape N by 24 hours
            vin=vin.reshape(-1,24)
            ein=np.delete(ewd,[7],0)
            ein=ein[ein!=0]
            ein=ein.reshape(-1,24)
            #stack
            xin=np.stack((vin,ein),2)#N*24*2
            #360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout=np.delete(gwd,[7],0)
            gout=gout[gout!=0]
            gout=gout.reshape(-1,24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin=vwd[[7],:]
            vin=vin[vin!=0]
            ein=ewd[[7],:]
            ein=ein[ein!=0]
            self.sqlen=int(sum(day[7,:]))
            vin=vin.reshape(-1,24)
            ein=ein.reshape(-1,24)
            xin=np.stack((vin,ein),2)
            #xin=xin.reshape(-1,24,2)
            #unsqueeze
            #xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout=gwd[[7],:]
            gout=gout[gout!=0]
            gout=gout.reshape(-1,24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            #self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx=idx[[7],:]
            self.idx=torch.tensor(self.idx, dtype=torch.float32)

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index,:]
            Y = self.Y[index,:]
        else:
            X = self.X[index,:]
            Y = self.Y[index,:]
        return X, Y

    def __len__(self):
        return self.len
class Karl_summer(Dataset):
    def __init__(self, train=True):
        self.train = train
        import h5py
        filename = './Karl.mat'
        with h5py.File(filename, 'r') as f:
            gtd= f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
        with h5py.File('./gacos_Karl_summer.mat', 'r') as f:
            gacosD400=f['gacosD400'][:]
            gacosD403=f['gacosD403'][:]
        if train:
            vin=np.delete(vwd,[5,7],0) #7*360
            #reshape N by 24 hours
            vin=vin.reshape(-1,24)
            ein=np.delete(ewd,[5,7],0)
            ein=ein.reshape(-1,24)
            #stack
            xin=np.stack((vin,ein),2)#N*24*2
            #360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout=np.delete(gwd,[5,7],0)
            gout=gout.reshape(-1,24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin=vwd[[5,7],:]
            ein=ewd[[5,7],:]
            vin=vin.reshape(-1,24)
            ein=ein.reshape(-1,24)
            xin=np.stack((vin,ein),2)
            #xin=xin.reshape(-1,24,2)
            #unsqueeze
            #xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout=gwd[[5,7],:]
            gout=gout.reshape(-1,24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            #self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx=idx[[5,7],:]
            self.idx=torch.tensor(self.idx, dtype=torch.float32)
            self.gacosD400=gacosD400-vhd[5,:]
            self.gacosD403=gacosD403-vhd[7,:]
        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index,:]
            Y = self.Y[index,:]
        else:
            X = self.X[index,:]
            Y = self.Y[index,:]
        return X, Y

    def __len__(self):
        return self.len
class Karl_GNSS(Dataset):
    def __init__(self,train=True):
        self.train=train
        import h5py
        filename = './GNSS.mat'
        with h5py.File(filename, 'r') as f:
            gpsdata = f['GPSDATA'][:]
            era5data = f['ERA5DATA'][:]
            era5data=era5data.T
        self.era5=torch.tensor(era5data[2,:-1], dtype=torch.float32).unsqueeze(1)
        if train:
            gpsdata=np.delete(gpsdata,2,0)
            era5data=np.delete(era5data,2,0)
            #self.X = torch.tensor(gpsdata[:,:-1], dtype=torch.float32)
            self.Y = torch.tensor(gpsdata, dtype=torch.float32)
            #self.Y = torch.tensor(gpsdata[:,1:], dtype=torch.float32)
            self.X = torch.tensor(era5data[:7, :], dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            #self.X = torch.tensor(gpsdata[2,:-1], dtype=torch.float32)
            #self.Y = torch.tensor(gpsdata[2,1:], dtype=torch.float32)
            self.X = torch.tensor(era5data[2, :], dtype=torch.float32)
            self.Y = torch.tensor(gpsdata[2, :], dtype=torch.float32)
            self.len = 1
        #unsqueeze
        self.X = self.X.unsqueeze(1)
        self.Y = self.Y.unsqueeze(1)
        #permute
        if train:
            self.X = torch.permute(self.X,(0,2,1))
            self.Y = torch.permute(self.Y,(0,2,1))
        # else:
        #     self.X = torch.permute(self.X,(1,0))
        #     self.Y = torch.permute(self.Y,(1,0))
        enstest=getensambledin()
        self.enstest=torch.tensor(enstest[:-1], dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index,:]
            Y = self.Y[index,:]
            return X, Y
        else:
            X = self.X
            Y = self.Y
            return X, Y

    def __len__(self):
        return self.len
class Niderland(Dataset):
    def __init__(self, train=True,season='winter'):
        self.train = train
        testsiteidx=3
        import h5py
        filename = './Niderland.mat'
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
        if season == 'winter':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        vwd = vwd * day

        ewd = ewd * day

        idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 7*360
            vin = vin[vin != 0]
            # reshape N by 24 hours
            vin = vin.reshape(-1, 24)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = ein[ein != 0]
            ein = ein.reshape(-1, 24)
            # stack
            xin = np.stack((vin, ein), 2)  # N*24*2
            # 360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin = vwd[testsiteidx, :]
            vin = vin[vin != 0]
            ein = ewd[testsiteidx, :]
            ein = ein[ein != 0]
            self.sqlen = int(sum(day[testsiteidx, :]))
            vin = vin.reshape(-1, 24)
            ein = ein.reshape(-1, 24)
            xin = np.stack((vin, ein), 2)
            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx = idx[testsiteidx, :]
            self.idx = torch.tensor(self.idx, dtype=torch.float32)

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index, :]
            Y = self.Y[index, :]
        else:
            X = self.X[index, :]
            Y = self.Y[index, :]
        return X, Y
    def __len__(self):
        return self.len
class Greenland(Dataset):
    def __init__(self, train=True, season='winter',testsiteidx=6):
        self.train = train
        #testsiteidx = 6
        import h5py
        filename = './Greenland.mat'
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
        if season == 'winter':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        vwd = vwd * day

        ewd = ewd * day

        idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 7*360
            vin = vin[vin != 0]
            # reshape N by 24 hours
            vin = vin.reshape(-1, 24)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = ein[ein != 0]
            ein = ein.reshape(-1, 24)
            # stack
            xin = np.stack((vin, ein), 2)  # N*24*2
            # 360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin = vwd[testsiteidx, :]
            vin = vin[vin != 0]
            ein = ewd[testsiteidx, :]
            ein = ein[ein != 0]
            self.sqlen = int(sum(day[testsiteidx, :]))
            vin = vin.reshape(-1, 24)
            ein = ein.reshape(-1, 24)
            xin = np.stack((vin, ein), 2)
            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx = idx[testsiteidx, :]
            self.idx = self.idx[day[testsiteidx, :] != 0]
            self.idx = torch.tensor(self.idx, dtype=torch.float32)

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index, :]
            Y = self.Y[index, :]
        else:
            X = self.X[index, :]
            Y = self.Y[index, :]
        return X, Y

    def __len__(self):
        return self.len
class France(Dataset):
    def __init__(self, train=True, season='winter', testsiteidx=0):
        self.train = train
        # testsiteidx = 6
        import h5py
        filename = './France.mat'
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
        if season == 'winter':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        vwd = vwd * day

        ewd = ewd * day

        idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 7*360
            vin = vin[vin != 0]
            # reshape N by 24 hours
            vin = vin.reshape(-1, 24)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = ein[ein != 0]
            ein = ein.reshape(-1, 24)
            # stack
            xin = np.stack((vin, ein), 2)  # N*24*2
            # 360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin = vwd[testsiteidx, :]
            vin = vin[vin != 0]
            ein = ewd[testsiteidx, :]
            ein = ein[ein != 0]
            self.sqlen = int(sum(day[testsiteidx, :]))
            vin = vin.reshape(-1, 24)
            ein = ein.reshape(-1, 24)
            xin = np.stack((vin, ein), 2)
            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx = idx[testsiteidx, :]
            self.idx = self.idx[day[testsiteidx, :] != 0]
            self.idx = torch.tensor(self.idx, dtype=torch.float32)

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index, :]
            Y = self.Y[index, :]
        else:
            X = self.X[index, :]
            Y = self.Y[index, :]
        return X, Y

    def __len__(self):
        return self.len
class Paris(Dataset):
    def __init__(self, train=True, season='winter', zone='15'):
        self.train = train
        if zone=='15':
            testsiteidx = 1
        elif zone=='50':
            testsiteidx = 7
        elif zone=='100':
            testsiteidx = 20
        elif zone=='150':
            testsiteidx = 17
        import h5py
        filename = './Paris'+zone+'.mat'
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
        if season == 'summer':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        vwd = vwd * day

        ewd = ewd * day

        idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 7*360
            vin = vin[vin != 0]
            # reshape N by 24 hours
            vin = vin.reshape(-1, 24)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = ein[ein != 0]
            ein = ein.reshape(-1, 24)
            # stack
            xin = np.stack((vin, ein), 2)  # N*24*2
            # 360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin = vwd[testsiteidx, :]
            vin = vin[vin != 0]
            ein = ewd[testsiteidx, :]
            ein = ein[ein != 0]
            self.sqlen = int(sum(day[testsiteidx, :]))
            vin = vin.reshape(-1, 24)
            ein = ein.reshape(-1, 24)
            xin = np.stack((vin, ein), 2)
            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx = idx[testsiteidx, :]
            self.idx = self.idx[day[testsiteidx, :] != 0]
            self.idx = torch.tensor(self.idx, dtype=torch.float32)

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index, :]
            Y = self.Y[index, :]
        else:
            X = self.X[index, :]
            Y = self.Y[index, :]
        return X, Y

    def __len__(self):
        return self.len
class Zone(Dataset):
    def __init__(self, train=True, season='winter', zone='1'):
        self.train = train
        testsiteidx = 0
        import h5py
        filename = './Francezone'+zone+'.mat'
        with h5py.File(filename, 'r') as f:
            gtd = f['gtd'][:]
            gwd = f['gwd'][:]
            vhd = f['vhd'][:]
            vwd = f['vwd'][:]
            ewd = f['ewd'][:]
            idx = f['idx'][:]
            idxday = f['idxday'][:]
        if season == 'winter':
            gtd = gtd[:, 0:360]
            gwd = gwd[:, 0:360]
            vhd = vhd[:, 0:360]
            vwd = vwd[:, 0:360]
            ewd = ewd[:, 0:360]
            day = idxday[:, 0:360]
            idx = idx[:, 0:360]
        else:
            gtd = gtd[:, 360:720]
            gwd = gwd[:, 360:720]
            vhd = vhd[:, 360:720]
            vwd = vwd[:, 360:720]
            ewd = ewd[:, 360:720]
            day = idxday[:, 360:720]
            idx = idx[:, 360:720]
        gtd = gtd * day

        gwd = gwd * day

        vhd = vhd * day

        vwd = vwd * day

        ewd = ewd * day

        idx = idx * day
        if train:
            vin = np.delete(vwd, testsiteidx, 0)  # 7*360
            vin = vin[vin != 0]
            # reshape N by 24 hours
            vin = vin.reshape(-1, 24)
            ein = np.delete(ewd, testsiteidx, 0)
            ein = ein[ein != 0]
            ein = ein.reshape(-1, 24)
            # stack
            xin = np.stack((vin, ein), 2)  # N*24*2
            # 360-> every 24 hours

            self.X = torch.tensor(xin, dtype=torch.float32)
            gout = np.delete(gwd, testsiteidx, 0)
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            self.len = self.X.shape[0]
        else:
            vin = vwd[testsiteidx, :]
            vin = vin[vin != 0]
            ein = ewd[testsiteidx, :]
            ein = ein[ein != 0]
            self.sqlen = int(sum(day[testsiteidx, :]))
            vin = vin.reshape(-1, 24)
            ein = ein.reshape(-1, 24)
            xin = np.stack((vin, ein), 2)
            # xin=xin.reshape(-1,24,2)
            # unsqueeze
            # xin=xin.unsqueeze(1)
            self.X = torch.tensor(xin, dtype=torch.float32)

            gout = gwd[testsiteidx, :]
            gout = gout[gout != 0]
            gout = gout.reshape(-1, 24)
            self.Y = torch.tensor(gout, dtype=torch.float32)
            # self.Y=self.Y.unsqueeze(0)
            self.len = self.X.shape[0]
            self.idx = idx[testsiteidx, :]
            self.idx = self.idx[day[testsiteidx, :] != 0]
            self.idx = torch.tensor(self.idx, dtype=torch.float32)

        self.Y = self.Y.unsqueeze(2)

    def __getitem__(self, index):
        if self.train:
            X = self.X[index, :]
            Y = self.Y[index, :]
        else:
            X = self.X[index, :]
            Y = self.Y[index, :]
        return X, Y

    def __len__(self):
        return self.len
class Karl_GNSS_dis(Dataset):
    def __init__(self,train=True):
        self.train=train
        import h5py
        filename = './GNSS.mat'
        with h5py.File(filename, 'r') as f:
            gpsdata = f['GPSDATA'][:]
            era5data = f['ERA5DATA'][:]
            era5data=era5data.T
        self.era5=torch.tensor(era5data[2,:-1], dtype=torch.float32).unsqueeze(1)
        if train:
            gpsdata=np.delete(gpsdata,2,0)
            miu=np.mean(gpsdata,axis=0)
            sigma=np.std(gpsdata,axis=0)
            self.X = torch.tensor(np.array([miu[:-1],sigma[:-1]]), dtype=torch.float32)
            self.Y = torch.tensor(np.array([miu[1:],sigma[1:]]), dtype=torch.float32)
            self.len = 1
        else:
            self.X = torch.tensor(gpsdata[2,:-1], dtype=torch.float32)
            self.Y = torch.tensor(gpsdata[2,1:], dtype=torch.float32)
            self.len = 1
        #unsqueeze
        #self.X = self.X.unsqueeze(1)#should be 359*2
        #self.Y = self.Y.unsqueeze(1)
        #permute
        #if train:
        #    self.X = torch.permute(self.X,(0,2,1))
        #    self.Y = torch.permute(self.Y,(0,2,1))
        self.X=self.X.T
        self.Y=self.Y.T
        # else:
        #     self.X = torch.permute(self.X,(1,0))
        #     self.Y = torch.permute(self.Y,(1,0))

    def __getitem__(self, index):
        if self.train:
            X = self.X
            Y = self.Y
            return X, Y
        else:
            X = self.X
            Y = self.Y
            return X, Y

    def __len__(self):
        return self.len

def getensambledin():
    gpsxyz = np.array([[4157306.847, 671172.249, 4774690.855], [4204025.405, 615170.792, 4741903.399],
                       [4179844.158, 667886.027, 4755620.578], [4165864.196, 719747.938, 4760759.510],
                       [4192084.452, 620213.795, 4751867.065], [4205647.851, 725930.847, 4724763.815],
                       [4222994.743, 633207.844, 4722898.293], [4146524.120, 613138.367, 4791517.374],
                       [4147317.850, 611392.595, 4791058.218]])
    dist = np.zeros((9, 9))
    for i in range(0, 9):
        for j in range(0, 9):
            dist[i, j] = np.sqrt(np.sum((gpsxyz[i, :] - gpsxyz[j, :]) ** 2))
    dis = dist[2, :-1]
    dis = np.delete(dis, 2)
    p = 1 / dis
    p = p / np.sum(p)
    import pickle
    with open('poutlist.pkl', 'rb') as f:
        poutlist = pickle.load(f)
    poutlist = np.array(poutlist)
    poutlist = poutlist.squeeze(1)
    ensumbledsq = np.dot(p, poutlist)
    return ensumbledsq

#main
if __name__ == '__main__':
    testid=5
    filepath='/home/wanduo/pyZTD/dataset/Tuebingen.mat'
    Tset=DatafromMat_LLH(filename=filepath,testsiteidx=testid,train=True,season='winter')
    #ramdom split
    trainset, testset = torch.utils.data.random_split(Tset, [int(0.8 * len(Tset)), len(Tset) - int(0.8 * len(Tset))])
    #testset=DatafromMat_LLH(filename=filepath,testsiteidx=testid,train=False,season='winter')
    print(len(trainset))
    print(len(testset))
