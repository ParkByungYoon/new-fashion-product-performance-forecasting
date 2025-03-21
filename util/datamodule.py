import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
import json
    

class BasicDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.train_item_ids = pickle.load(open(os.path.join(args.data_dir, 'train_item_ids.pkl'), 'rb'))
        self.valid_item_ids = pickle.load(open(os.path.join(args.data_dir, 'valid_item_ids.pkl'), 'rb'))
        self.test_item_ids = pickle.load(open(os.path.join(args.data_dir, 'test_item_ids.pkl'), 'rb'))
    
    def prepare_data(self):
        self.data_dict = json.load(open(os.path.join(self.data_dir, "data.json"), "r"))
        self.data_dict['image_embedding'] = pickle.load(open(os.path.join(self.data_dir, "fclip_image.pkl"), "rb"))
        self.data_dict['text_embedding'] = pickle.load(open(os.path.join(self.data_dir, "fclip_text.pkl"), "rb"))

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = BasicDataset(self.args, self.data_dict, self.train_item_ids)
            self.valid_dataset = BasicDataset(self.args, self.data_dict, self.valid_item_ids)
        if stage == "test":
            self.test_dataset = BasicDataset(self.args, self.data_dict, self.test_item_ids)
        if stage == "predict":
            self.test_dataset = BasicDataset(self.args, self.data_dict, self.test_item_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
    

class BasicDataset(Dataset):
    def __init__(self, args, data_dict, item_ids):
        super().__init__()
        self.use_trend = args.use_trend
        self.use_weather = args.use_weather
        self.use_meta_sale = args.use_meta_sale
        self.data_dict = data_dict
        self.item_ids = item_ids
        self.__preprocess__()

    def __preprocess__(self):
        item_ids, item_sales, endo_inputs, exo_inputs, release_dates, image_embeddings, text_embeddings, meta_data  = [[] for _ in range(8)]

        for item_id in tqdm(self.item_ids, total=len(self.item_ids), ascii=True):
            sales = self.data_dict[item_id]['item_sales']
            release_date = self.data_dict[item_id]['release_date']
            img_emb = self.data_dict['image_embedding'][item_id] if item_id in self.data_dict['image_embedding'] else np.random.normal(size=512).tolist()
            txt_emb = self.data_dict['text_embedding'][item_id] if item_id in self.data_dict['text_embedding'] else np.random.normal(size=512).tolist()
            meta = self.data_dict[item_id]['meta_data']
            
            exo = []
            if self.use_trend: exo.extend(self.data_dict[item_id]['trend'])
            if self.use_weather: exo.extend(self.data_dict[item_id]['weather'])
            if self.use_meta_sale: exo.extend(self.data_dict[item_id]['meta_sale'])

            endo = self.data_dict[item_id]['endo_vars']
            
            item_sales.append(sales)
            release_dates.append(release_date)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            meta_data.append(meta)
            endo_inputs.append(endo)
            exo_inputs.append(exo)
            item_ids.append(item_id)
        
        self.item_ids = item_ids
        self.item_sales = torch.FloatTensor(np.array(item_sales))
        self.endo_inputs = torch.FloatTensor(np.array(endo_inputs))
        self.exo_inputs = torch.FloatTensor(np.array(exo_inputs))
        self.release_dates = torch.FloatTensor(np.array(release_dates))
        self.image_embeddings = torch.FloatTensor(np.array(image_embeddings))
        self.text_embeddings = torch.FloatTensor(np.array(text_embeddings))
        self.meta_data = torch.FloatTensor(np.array(meta_data))
    
    def __getitem__(self, idx):
        return \
            self.item_sales[idx], \
            self.endo_inputs[idx],\
            self.exo_inputs[idx],\
            self.release_dates[idx],\
            self.image_embeddings[idx],\
            self.text_embeddings[idx],\
            self.meta_data[idx], \

    def __len__(self):
        return len(self.item_ids)

class VisuelleDataModule(BasicDataModule):
    def __init__(self, args):
        args.use_trend = True
        args.use_weather = args.num_exo_vars in [5,8]
        args.use_meta_sale = args.num_exo_vars in [6,8]
        self.meta_cols = ['category',  'color', 'fabric']
        super().__init__(args)

class MindBridgeDataModule(BasicDataModule):
    def __init__(self, args):
        args.use_trend = True
        args.use_weather = args.num_exo_vars in [5,9]
        args.use_meta_sale = args.num_exo_vars in [7,9]
        self.meta_cols = ['brand', 'category',  'color', 'fabric']
        super().__init__(args)