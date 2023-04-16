import numpy as np
import pandas as pd

import pathlib
import string
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

import torch

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

from datasets.base import BaseDataset

def sample_from_letter(fnm_list, n_items, case = None):
    #case: ('lower','upper')
    df = pd.DataFrame()
    df['fname'] = [fnm.split('.')[0] for fnm in fnm_list]
    spl = df.fname.str.split('_')
    df['letter'] = spl.apply(lambda x: x[0])
    df['case'] = spl.apply(lambda x: x[-1])
    df['class'] = df['letter']+df['case']
    df['class'] = pd.Categorical(df['class'])
    df['label'] = df['class'].cat.codes    

    n_classes = df['class'].nunique()
    
    if case is not None:
        df = df[df.case == case]
    samples = df.groupby('label').apply(lambda x: x.sample(n_items))
    fnm_labels = samples[['fname','label']].set_index('fname').label.to_dict()
    return n_classes, fnm_labels

class RankingDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return self.num_classes

    def __init__(
        self,
        root_dir,
        fnm_labels,
        num_classes,
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Args:
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        # path = pathlib.Path(root_dir)
        self.random_rotate = random_rotate
        self.num_classes = num_classes
        
        self.lbs = fnm_labels

        file_paths = [pathlib.Path(root_dir+fnm+'.bin') for fnm in fnm_labels.keys()]
        print(file_paths[0], file_paths[0].exists())
        self.load_graphs(file_paths, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally get the label from the filename and store it in the sample dict

        sample["label"] = torch.tensor([self.lbs[str(file_path.stem)]]).long()
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        return collated
    
def encode(model, loader, device):
    embs_list = []
    labels_list = []
    with torch.no_grad():  
        for batch in loader:
            inputs = batch["graph"].to(device)
            inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
            inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
            embs_list.append(model.encode_part(inputs).to(device=torch.device('cpu')))
                        
            labels_list.append(batch["label"].to(device=torch.device('cpu')))
    return embs_list, labels_list

def cals_map_all(test_loaders, model, device):
    model = model.eval()
    metr = []
    for loader in test_loaders:
        e_list, l_list = encode(model, loader, device)
        embs = torch.cat(e_list,dim=0).numpy()
        lbs = torch.cat(l_list,dim=0).numpy()
        metr.append(calc_map(embs, lbs))
    return np.mean(metr)

def calc_map(X, labels, K = 5):
    tree = KDTree(X, leaf_size=40)  # creating kd tree
    _, ind = tree.query(X, k=K+1)  # quering nearest items

    is_valid_label = (labels[ind[:,1:]] == labels.reshape(-1,1)).astype(int)

    cum_sum = np.cumsum(is_valid_label, axis=1)
    P_K = cum_sum/np.arange(1, K+1).reshape(1,-1)
    AP_K = P_K.sum(axis=1) / np.clip(cum_sum[:,-1],1, K)

    return AP_K.mean()
