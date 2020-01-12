from torch.utils import data
import os
from scipy.io import loadmat
import numpy as np
import cv2
import torch as t


class MySet(data.Dataset):

    def __init__(self, image_path, mask_path):
        names = [i.strip(".jpg") for i in os.listdir(image_path)]
        self.image_paths = [os.path.join(image_path, i + ".jpg") for i in names]
        self.mask_paths = [os.path.join(mask_path, i + ".mat") for i in names]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        mask = loadmat(mask_path)["groundTruth"][0][0][1].astype(np.int16)
        image = cv2.imread(image_path).astype(np.int16)
        image, mask = self.data_process(image, mask)
        return t.tensor(image).type(t.FloatTensor), t.tensor(mask).type(t.LongTensor)

    def __len__(self):
        return len(self.image_paths)

    def data_process(self, image, mask):
        image = cv2.resize(image, (388, 388))
        image = np.transpose(cv2.copyMakeBorder(image, 92, 92, 92, 92, cv2.BORDER_REFLECT), [2, 0, 1])
        mask = cv2.resize(mask, (388, 388), interpolation=cv2.INTER_NEAREST)
        return image, mask


def make_loader(image_path, mask_path, batch_size):
    return iter(data.DataLoader(MySet(image_path, mask_path), batch_size=batch_size, shuffle=True, drop_last=False))
