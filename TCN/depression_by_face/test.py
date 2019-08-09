import os
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

data_root_dir = '/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/csvdata/'
data_300_2d = '300_CLNF_features.csv'
label_file = '/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/label2D.csv'
# def data_generator(args):
#     features = pd.read_csv(data_root_dir + data_300_2d)
#     print(features)


# features = pd.read_csv("/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/output2D/303/24.csv")
# print(features)
#
# n = 3 + 68 + 68
# array = features.iloc[:, 4:]
# print('read area')
# print(array)
# matrix1 = array.values.astype('float')
# print(matrix1)
#
# features_variable = Variable(torch.from_numpy(matrix1))
# print(features_variable)

# print(np.zeros([136], dtype=float))
print(np.zeros(136))