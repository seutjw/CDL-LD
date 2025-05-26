import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class Multi_view_data(Dataset):


    def __init__(self, root, train=True):

        super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)
        view_number = 1
        self.X = dict()
        if train:
            self.X[0] = dataset['X_tr']
            y = dataset['Dr_tr']
        else:
            self.X[0] = dataset['X_test']
            y = dataset['Dr_test']

        self.y = y

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target


    def __len__(self):
        return len(self.X[0])

    def lenx(self):
        return len(self.X[0][0])

    def leny(self):
        return len(self.y[0])


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x
