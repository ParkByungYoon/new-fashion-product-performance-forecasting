import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle

class MultiVariateDataset(Dataset):
    def __init__(self, data_dict, item_ids):
        super().__init__()
        self.data_dict = data_dict
        self.item_ids = item_ids
        self.__preprocess__()

    def __preprocess__(self):
        item_sales, release_indices, release_dates, image_embeddings, text_embeddings, meta_data = [[] for _ in range(6)]
        item_ids = []

        for item_id in tqdm(self.item_ids, total=len(self.item_ids), ascii=True):
            if item_id not in self.data_dict['sales'].index: continue
            if item_id not in self.data_dict['release_date'].index: continue
            if item_id not in self.data_dict['meta'].index: continue
            
            sales = self.data_dict['sales'].loc[item_id].values
            release_idx = self.data_dict['release_idx'][item_id]
            release_date = self.data_dict['release_date'].loc[item_id].values
            
            if item_id not in self.data_dict['image_embedding']:
                img_emb = np.random.normal(size=512).tolist()
            else: 
                img_emb = self.data_dict['image_embedding'][item_id]
            
            if item_id[:-2] not in self.data_dict['text_embedding']:
                txt_emb = np.random.normal(size=512).tolist()
            else: 
                txt_emb = self.data_dict['text_embedding'][item_id[:-2]]

            meta = self.data_dict['meta'].loc[item_id].values
            
            item_sales.append(sales)
            release_indices.append(release_idx)
            release_dates.append(release_date)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            meta_data.append(meta)
            item_ids.append(item_id)
        
        self.item_ids = item_ids
        self.item_sales = torch.FloatTensor(np.array(item_sales))

        self.multi_vars = torch.FloatTensor(self.data_dict['multi_vars'].values)

        self.release_indices = torch.FloatTensor(np.array(release_indices))
        self.release_dates = torch.FloatTensor(np.array(release_dates))
        self.image_embeddings = torch.FloatTensor(np.array(image_embeddings))
        self.text_embeddings = torch.FloatTensor(np.array(text_embeddings))
        self.meta_data = torch.FloatTensor(np.array(meta_data))
    
    def __getitem__(self, idx):
        multi_vars = self.multi_variate(idx)
        multi_vars[:27] = self.normalize(multi_vars[:27])
        multi_vars[27:34] = self.normalize(multi_vars[27:34])
        multi_vars[34:] = self.normalize(multi_vars[34:])
        return \
            self.item_sales[idx], multi_vars,\
            self.release_dates[idx],\
            self.image_embeddings[idx],\
            self.text_embeddings[idx],\
            self.meta_data[idx]

    def multi_variate(self, idx):
        release_idx = self.release_indices[idx].to(torch.int64).item()
        if release_idx - 52 < 0:
            zero_pad = torch.zeros(self.multi_vars.size(0), 52)
            zero_pad[:, 52-release_idx:52] = self.multi_vars[:,:release_idx]
            return zero_pad
        return self.multi_vars[:,release_idx-52:release_idx]

    def normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-5)

    def __len__(self):
        return len(self.item_ids)


class MultiVariateDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.mu = args.mu
        self.sigma = args.sigma
        self.batch_size = args.batch_size
        self.train_item_ids = pickle.load(open(os.path.join(args.data_dir, 'train_item_ids.pkl'), 'rb'))
        self.valid_item_ids = pickle.load(open(os.path.join(args.data_dir, 'valid_item_ids.pkl'), 'rb'))
        self.test_item_ids = pickle.load(open(os.path.join(args.data_dir, 'test_item_ids.pkl'), 'rb')).tolist()
    
    def prepare_data(self):
        self.data_dict = {}
        self.data_dict['multi_vars'] = pd.read_csv(os.path.join(self.data_dir, 'multi_vars.csv'), index_col=0)
        self.data_dict['sales'] = (pd.read_csv(os.path.join(self.data_dir, 'sales.csv'), index_col=0).iloc[:,1:] - self.mean) / self.std
        self.data_dict['release_idx'] = pickle.load(open(os.path.join(self.data_dir, 'release_idx.pkl'), 'rb'))
        self.data_dict['release_date'] = pd.read_csv(os.path.join(self.data_dir, 'release_date.csv'), index_col=0)
        self.data_dict['meta'] = pd.read_csv(os.path.join(self.data_dir, 'meta.csv'), index_col=0).sort_index(axis=1)
        self.data_dict['image_embedding'] = pickle.load(open(os.path.join(self.data_dir, 'image_embedding.pkl'), 'rb'))
        self.data_dict['text_embedding'] = pickle.load(open(os.path.join(self.data_dir, 'text_embedding.pkl'), 'rb'))

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MultiVariateDataset(self.data_dict, self.train_item_ids)
            self.valid_dataset = MultiVariateDataset(self.data_dict, self.valid_item_ids)
        if stage == "test":
            self.test_dataset = MultiVariateDataset(self.data_dict, self.test_item_ids)
        if stage == "predict":
            self.test_dataset = MultiVariateDataset(self.data_dict, self.test_item_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)