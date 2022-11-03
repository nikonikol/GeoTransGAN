import torch
import numpy as np
import torchvision
import torchvision.transforms as T
import scipy.io as io
from torch.utils.data import Dataset, Sampler

class CrypkoDataset(Dataset):
    def __init__(self, fnames):
        # self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # 1. Load the image
        # img = torchvision.io.read_image(fname)
        img = io.loadmat(fname)['sample'][0]
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        # 2. Resize and normalize the images using torchvision.
        # img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.N = len(data_source)


    def __iter__(self):
        while True:
            for idx in torch.randperm(self.N):
                yield idx

def handle_3Ddata(trainning_data_path,label_data_path,trainning_data_name,training_data_format,label_data_format,standard_x,standard_y,standard_z,RGB_data = False):#直接将输入的数据变为单个文件的路径，批量的处理写在外面

    trainning_data_path = trainning_data_path + trainning_data_name + training_data_format

    trainning_data_arr = read_3Dfile(trainning_data_path)
    mask = np.zeros(trainning_data_arr.shape)
    mask[:,12,:] = 1#数据掩膜
    mask[:,24,:] = 1
    mask[:,:,12] = 1
    mask[:,:,24] = 1
    label_data_arr = trainning_data_arr#标签
    trainning_data_arr=mask*trainning_data_arr#条件数据

    if RGB_data == False:
        trainning_data_arr = np.expand_dims(np.array(trainning_data_arr).astype(np.float32),axis=3)
        trainning_data_arr = np.concatenate([trainning_data_arr, trainning_data_arr, trainning_data_arr], axis=3)
        label_data_arr = np.expand_dims(np.array(label_data_arr).astype(np.float32),axis=3)
        label_data_arr = np.concatenate([label_data_arr, label_data_arr, label_data_arr], axis=3)

    #进行归一化处理（便于收敛）先归到0-1，然后调整到-1到1
    trainning_data_arr = trainning_data_arr  / 5. - 1
    label_data_arr = label_data_arr / 5. - 1

    return trainning_data_arr,label_data_arr