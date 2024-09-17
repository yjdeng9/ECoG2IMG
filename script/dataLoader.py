

from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class single_IFP(Dataset):
    def __init__(self, subj_id, ifp_dir, img_dir, time_len = 512,
                 meta_info=None, response_type='all'):

        if meta_info is None:
            meta_info = pd.read_csv('../../data/meta_info.csv')
        if response_type == 1:
            meta_info = meta_info[meta_info['response'] == 1]
        elif response_type == 0:
            meta_info = meta_info[meta_info['response'] == 0]

        self.subj_df = meta_info[meta_info['subj_id'] == subj_id]
        self.subj_df = self.subj_df.reset_index(drop=True)
        self.subj_id = subj_id
        self.ifp_dir = ifp_dir
        self.img_dir = img_dir

        self.time_len = time_len

        self.category_dict = {1:'animal', 2:'chair', 3:'face', 4:'fruit',5:'lego',6:'shoe', 7:'vehicle'}
        self.label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


    def __len__(self):
        return len(self.subj_df)

    def norm_time_ifp(self, signal):
        norm_len = self.time_len

        n_channels = signal.shape[0]
        n_time = signal.shape[1]
        if n_time == norm_len:
            return signal
        else:
            norm_signal = np.zeros((n_channels, norm_len))
            for c in range(n_channels):
                signal_c = signal[c, :]
                interp_func = interp1d(np.linspace(0, 1, len(signal_c)), signal_c, kind='linear')
                norm_signal[c, :] = interp_func(np.linspace(0, 1, norm_len))

        return norm_signal

    def transform_img(self, x):
        transform_er = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        return transform_er(x)

    def transform_ifp(self, x):
        # x = x.transpose((1,0))
        x = torch.Tensor(x.copy())
        x = x.unsqueeze(0)
        return x

    def __getitem__(self, idx):
        subj_id = self.subj_df.loc[idx, 'subj_id']
        trial_idx = int(self.subj_df.loc[idx, 'trial_idx'])
        image_idx = self.subj_df.loc[idx, 'image_index']
        response = self.subj_df.loc[idx, 'response']
        category = self.subj_df.loc[idx, 'category']
        rotation = self.subj_df.loc[idx, 'rotation']
        category_name = self.category_dict[category]
        label = self.label_map[category]

        signal_path = os.path.join(self.ifp_dir, '%s_trial%d.npy' % (subj_id, trial_idx))
        signal = np.load(signal_path)
        signal = self.norm_time_ifp(signal)
        signal = self.transform_ifp(signal)

        img_path = os.path.join(self.img_dir, 'image_%d.npy' % (image_idx))
        img = np.load(img_path)
        img = img.astype(np.float32)

        return signal, img, label, category_name, rotation, response



def main():
    dataset = single_IFP(subj_id='subj12', data_dir='')
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=1)
    # for i, data in enumerate(dataloader):
    #     print(i, data)
    #     break


if __name__ == '__main__':
    main()

