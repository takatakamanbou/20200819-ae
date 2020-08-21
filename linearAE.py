import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import LFWDataset
import ipytools

### definition of the network
#
class NN(nn.Module):

    def __init__(self, D=100, H=10):

        super(NN, self).__init__()
        # linear network => bias is not used
        self.fc1 = nn.Linear(D, H, bias=False)
        self.fc2 = nn.Linear(H, D, bias=False)

    def forward(self, X):
        Y = self.fc1(X)
        Z = self.fc2(Y)
        return Z


if __name__ == '__main__':

    torch.manual_seed(0)

    ### device
    #
    use_gpu_if_available = True
    if use_gpu_if_available and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('# using', device)

    ### preparing the data (L)
    #
    dsL = LFWDataset.LFWDataset(LT='L')
    NL = dsL.ndat
    D = dsL.ndim
    print(NL, D)
    batchsize = 32
    dlL = torch.utils.data.DataLoader(dsL, batch_size=batchsize, shuffle=True, drop_last=True)
    nbatch = len(dlL)

    ### initializing the network
    #
    H = 100
    network = NN(D=D, H=H)
    model = network.to(device)
    print(model)
    #optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    optimizer = optim.Adam(model.parameters())
    print(optimizer)

    ### learning
    #
    nepoch = 100

    for i in range(nepoch):

        model.train()

        sqeL = np.empty(nbatch)

        for ib, Xb in enumerate(dlL):

            optimizer.zero_grad()
            Z = model(Xb)
            loss = F.mse_loss(Xb, Z, reduction='sum')
            loss.backward()
            optimizer.step()

            sqeL[ib] = loss.clone().cpu().detach().numpy()
            
        msqeL = np.sum(sqeL) / (NL*D)
        print(i, msqeL)


    ### evaluation (L)
    #
    model.eval()

    for ib, Xb in enumerate(dlL):
        Z = model(Xb)
        loss = F.mse_loss(Xb, Z, reduction='sum')
        sqeL[ib] = loss.clone().cpu().detach().numpy()

    msqeL = np.sum(sqeL) / (NL*D)

    print(f'# L: {msqeL}')

    ### preparing the data (T)
    #
    dsT = LFWDataset.LFWDataset(LT='T')
    NT = dsT.ndat
    print(NT, D)
    dlT = torch.utils.data.DataLoader(dsT, batch_size=batchsize, shuffle=False, drop_last=False)

    ### evaluation & reconstruction (T)
    #
    model.eval()
    sqeT = np.empty(len(dlT))
    for ib, Xb in enumerate(dlT):
        Z = model(Xb)
        loss = F.mse_loss(Xb, Z, reduction='sum')
        sqeT[ib] = loss.clone().cpu().detach().numpy()
        if ib == 0:
            ZT = Z.cpu().detach().numpy()
            ZT *= 255
            ZT += dsT.meanL.reshape(-1)
            ZT = np.clip(ZT, 0, 255)

    msqeT = np.sum(sqeT) / (NT*D)
    print(f'# T: {msqeT}')

    img = ipytools.mosaicImage(ZT[:16, ::], 8, 2)
    cv2.imwrite('hoge.png', img)

