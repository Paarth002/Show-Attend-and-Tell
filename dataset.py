import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform

        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_indices = json.load(open(data_path + '/{}_img_indices.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))

        self.qcnn_embed = np.load('/kaggle/input/qcnn-224-224-padded/qcnn-224-224-padded.npy')

    def __getitem__(self, index):
        # img_path = self.img_paths[index]
        # img = pil_loader(img_path)
        img = Image.fromarray(self.qcnn_embed[self.img_indices[index]]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.split_type == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[index])

        # matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        # all_captions = [self.captions[idx] for idx in matching_idxs]
        all_captions = [self.captions[index]]
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor(all_captions)

    def __len__(self):
        return len(self.captions)
