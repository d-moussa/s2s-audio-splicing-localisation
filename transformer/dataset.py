from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import pickle
import json

class PickleDataset(Dataset):
    def __init__(self, spec_pickle_dict, annotation_file, package_size, is_cnn=False, cnn_multisplice=False):
        
        with open(annotation_file, "r") as fp:
                dataset_dict = json.load(fp)
                dataset_dict = list(dataset_dict.values())
        df = pd.DataFrame(dataset_dict)
        self.labels = df["splice_bins"]
        self.labels = np.asarray(self.labels).reshape(-1)
        self.spec_pickle_dict = spec_pickle_dict
        self.specs = pickle.load(open(spec_pickle_dict[0], "rb"))
        self.package_size = package_size
        self.currently_loaded = 0
        self.is_cnn = is_cnn
        self.cnn_multisplice = cnn_multisplice
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        self.check_and_load_next_pickle(idx)
        label = self.labels[idx]
        spec = self.specs[idx-self.currently_loaded]
        spec = self.scale_min_max(spec, feature_range=(-1,1))

        if self.is_cnn:
            temp = torch.zeros(95, 256)
            max_cut = np.min([spec.shape[0], 95])
            temp[:max_cut, :] = spec[:max_cut, :]
            spec = temp
            spec = torch.unsqueeze(spec, 0)

            if self.cnn_multisplice:
                if len(label) < 5:
                    label += np.zeros((5-len(label),)).tolist()

        
        return spec, label

    def check_and_load_next_pickle(self, idx):
        if (idx - self.currently_loaded < 0) or (idx - self.currently_loaded >= self.package_size):
            self.currently_loaded = (int(idx/self.package_size))*self.package_size
            self.specs = pickle.load(open(self.spec_pickle_dict[self.currently_loaded], "rb"))

    def scale_min_max(self, tensor, feature_range=(-1,1)):
        a, b = feature_range

        dist = tensor.max() - tensor.min()
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min())
        tensor.mul_(b - a).add_(a)
        return tensor



class PickleDatasetMultiinput(Dataset):
    def __init__(self, spec_pickle_dict, annotation_file, package_size, is_cnn=False, cnn_multisplice=False):
        
        with open(annotation_file, "r") as fp:
                dataset_dict = json.load(fp)
                dataset_dict = list(dataset_dict.values())
        df = pd.DataFrame(dataset_dict)
        self.labels = df["splice_bins"]
        self.labels = np.asarray(self.labels).reshape(-1)
        self.spec_pickle_dict = spec_pickle_dict
        self.specs = pickle.load(open(spec_pickle_dict[0], "rb"))
        self.package_size = package_size
        self.currently_loaded = 0
        self.is_cnn = is_cnn
        self.cnn_multisplice = cnn_multisplice
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        self.check_and_load_next_pickle(idx)
        label = self.labels[idx]
        specs = []
        for s in self.specs:
            s = self.scale_min_max(s[idx-self.currently_loaded], feature_range=(-1,1))
            if s.shape[-1] == 1:
                s = s.repeat(1, 3)

            if self.is_cnn:
                temp = torch.zeros(95, s.shape[-1])
                max_cut = np.min([s.shape[0], 95])
                temp[:max_cut, :] = s[:max_cut, :]
                s = temp
                s = torch.unsqueeze(s, 0)

                if self.cnn_multisplice:
                    if len(label) < 5:
                        label += np.zeros((5-len(label),)).tolist()

            specs.append(s)
        if self.is_cnn:
            specs = torch.cat(specs, dim=-1)
        else:
            specs = torch.cat(specs, dim=1)
        return specs, label

    def check_and_load_next_pickle(self, idx):
        if (idx - self.currently_loaded < 0) or (idx - self.currently_loaded >= self.package_size):
            self.currently_loaded = (int(idx/self.package_size))*self.package_size
            self.specs = pickle.load(open(self.spec_pickle_dict[self.currently_loaded], "rb"))

    def scale_min_max(self, tensor, feature_range=(-1,1)):
        a, b = feature_range

        dist = tensor.max() - tensor.min()
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min())
        tensor.mul_(b - a).add_(a)
        return tensor