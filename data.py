import numpy as np
import os.path
import pickle
import cv2


class Data:

    def __init__(self, pathStr='./data/lfw-selected/'):

        self.path = os.path.normpath(pathStr)
        assert os.path.isdir(self.path)
        print(f'# {self.path}')

        with open(os.path.join(self.path, 'L/attributes.pickle'), 'rb') as f:
            rv = pickle.load(f)
        self.attrListL = rv['list']
        self.NL = len(self.attrListL)

        with open(os.path.join(self.path, 'T/attributes.pickle'), 'rb') as f:
            rv = pickle.load(f)
        self.attrListT = rv['list']
        self.NT = len(self.attrListT)

        print(f'# NL = {self.NL}, NT = {self.NT}')


    def get(self, LT='L'):

        assert LT == 'L' or LT == 'T'
        print(f'# reading \'{LT}\'...', end=' ')
        if LT == 'L':
            N = self.NL
        else:
            N = self.NT

        fn = os.path.join(self.path, LT, 'img0000.png')
        img = cv2.imread(fn)

        X = np.empty((N,) + img.shape)
        for i in range(N):
            fn = os.path.join(self.path, LT, f'img{i:04d}.png')
            X[i, ::] = cv2.imread(fn)

        print(X.shape)

        return X




if __name__ == '__main__':
    
    d = Data()
    XL = d.get('L')
    XT = d.get('T')
    print(XL.shape, XT.shape)

    img = np.mean(XL, axis=0)
    cv2.imwrite('meanL.png', img)
    img = np.mean(XT, axis=0)
    cv2.imwrite('meanT.png', img)