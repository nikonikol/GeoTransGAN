import torch
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