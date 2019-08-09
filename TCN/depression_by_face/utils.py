import os
import torch
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

data_root_dir = '/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/csvdata/'
data_300_2d = '300_CLNF_features.csv'
label_file = '/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/label2D.csv'

# # def data_generator(args):
# #     features = pd.read_csv(data_root_dir + data_300_2d)
# #     print(features)
#
# features = pd.read_csv(data_root_dir + data_300_2d)
# print(features)
#
# n = 3 + 68 + 68
# array = features.iloc[1:3, 1:68]
# print(array)
# matrix1 = array.values.astype('float')
# print(matrix1)
#
# features_variable = Variable(torch.from_numpy(matrix1))
# print(features_variable)
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import os

'/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/label2D.csv'
label2D_originFIle = pd.read_csv(label_file)
print(label2D_originFIle.info())
print(label2D_originFIle.head())

depression_level = label2D_originFIle['depression_level']
print(depression_level)
depression_level_matrix = Series.as_matrix(depression_level)
print(depression_level_matrix.shape)

# 查看抑郁症的种类
depression_set = set(depression_level_matrix)
print(len(depression_set))

# 划分训练集和测试集
files = Series.as_matrix(label2D_originFIle['filename'])
dir = '/Users/WeiJoseph/mystuff/ihappy_lifeifei/handle/'
i = 0
length = len(files)
while (i < length):
    files[i] = os.path.join(dir, files[i])
    i += 1
file_train = files[:8000]
file_test = files[8000:]

# np.save('file_train.npy', file_train)
# np.save('file_test.npy', file_test)

labels = Series.as_matrix(label2D_originFIle['depression_level'])
print(labels.shape)
label_train = labels[:8000]
label_test = labels[8000:]
# np.save('label_train.npy', label_train)
# np.save('label_test.npy', label_test)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

MAX_FRAME = 300


# def default_loader(path):
#     data = pd.read_csv(path, header=None)
#     pure_data = data.iloc[:, 4:]
#     data_length = pure_data.shape[0]
#     blank_row = np.zeros(136)
#     blank_dataframe_row = pd.DataFrame(blank_row)
#     # print("当前的视频帧数为" + str(data_length))
#     while data_length < 300:
#         # pure_data.append(blank_dataframe_row)
#         pure_data.loc[data_length] = blank_row
#         data_length += 1
#     if data_length > 300:
#         pure_data = pure_data.iloc[:300, :]
#     # print(pure_data.shape)
#     # print(pure_data)
#     return torch.tensor(pure_data.values)


def default_loader(path):
    data = pd.read_csv(path, header=None)
    pure_data = data.iloc[:, 4:]
    np_data = np.array(pure_data.values)

    if np_data.shape[0] < 300:
        redundant_data = np.zeros([300 - np_data.shape[0], 136])
        np_data = np.vstack((np_data, redundant_data))

    if np_data.shape[0] > 300:
        np_data = np_data[:300]
    print(np_data.shape)

    return torch.tensor(np_data)

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.input = file_train
        # print(file_train)
        self.target = label_train
        self.loader = loader

    def __getitem__(self, index):
        path = self.input[index]
        # print(str(index) + path)
        inputTensor = self.loader(path)
        target = self.target[index]
        return inputTensor, target

    def __len__(self):
        return len(self.input)


class testset(Dataset):
    def __init__(self, loader=default_loader):
        self.input = file_test
        self.target = label_test
        self.loader = loader

    def __getitem__(self, index):
        path = self.input[index]
        inputTensor = self.loader(path)
        target = self.target[index]
        return inputTensor, target

    def __len__(self):
        return len(self.input)
