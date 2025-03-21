import sys
sys.path.append('../')

import torch
import argparse
import numpy as np
import wandb
import random
import pandas as pd

import pytorch_lightning as pl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os
import importlib


def random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.autograd.set_detect_anomaly(True)

def run(args):
    args.data_dir = args.data_dir + f"/{args.dataset_name}"
    args.log_dir = args.log_dir + f"/{args.dataset_name}"
    args.result_dir = args.result_dir + f"/{args.dataset_name}"

    if args.dataset_name == 'MindBridge':
        args.center = 14.0
        args.scale = 37.0
    elif args.dataset_name == 'Visuelle':
        args.center = 0.0
        args.scale = 875.0

    print(args)
    random_seed(args.seed)

    model_module = importlib.import_module(f"MVTSF.model.{args.model_name}")
    model_cls = getattr(model_module, args.model_name)
    model = model_cls(args)

    dataset_module = importlib.import_module("MVTSF.util.datamodule")
    dataset_cls = getattr(dataset_module, f"{args.dataset_name}DataModule")
    dataset = dataset_cls(args)

    trainer = pl.Trainer(
        devices=[args.gpu_num],
        logger=False,
    )
    ckpt_path = os.path.join(args.log_dir, args.model_name, args.ckpt_name)
    prediction = trainer.predict(model, ckpt_path=ckpt_path, datamodule=dataset)

    result_df = pd.DataFrame(torch.cat(prediction, dim=0).numpy(), index=dataset.test_dataset.item_ids)
    result_df.to_csv(os.path.join(args.result_dir, f"{args.ckpt_name}".replace("ckpt","csv")))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate-Time-Series-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='/SSL_NAS/SFLAB/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    parser.add_argument('--model_name', type=str, default='Transformer')
    parser.add_argument('--dataset_name', type=str, default='MindBridge')
    parser.add_argument('--ckpt_name', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--endo_input_len', type=int, default=12)
    parser.add_argument('--exo_input_len', type=int, default=52)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    # Specific arguments
    parser.add_argument("--use_trend", action="store_true")
    parser.add_argument("--use_weather", action="store_true")
    parser.add_argument("--use_meta_sale", action="store_true")
    parser.add_argument('--segment_len', type=int, default=4)
    parser.add_argument('--num_endo_vars', type=int, default=4)
    parser.add_argument('--num_exo_vars', type=int, default=48)
    parser.add_argument("--num_meta", type=int, default=52)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Multivariate-Time-Series-Forecasting')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)