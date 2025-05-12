import torch
import argparse
import numpy as np
import wandb
import random
import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
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

    if not args.use_revin and args.dataset_name == 'TBH':
        args.center, args.scale = 40.89353961781402, 74.34192367524047
    elif not args.use_revin and args.dataset_name == 'Visuelle':
        args.center, args.scale  = 0.0, 875.0

    print(args)
    random_seed(args.seed)

    model_module = importlib.import_module(f"model.{args.model_name}")
    model_cls = getattr(model_module, args.model_name)
    model = model_cls(args)

    dataset_module = importlib.import_module("util.datamodule")
    dataset_cls = getattr(dataset_module, f"{args.dataset_name}DataModule")
    dataset = dataset_cls(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.log_dir,args.model_name),
        filename=f'{args.model_name}',
        monitor='valid_rescaled_adjusted_smape',
        mode='min',
        save_top_k=1
    )

    wandb.require("core")
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_proj +'-'+ args.dataset_name, 
        name=f'{args.model_name}-{datetime.datetime.now().strftime("%y%m%d-%H%M")}',
        dir=args.wandb_dir
    )
    wandb_logger = pl_loggers.WandbLogger()
    trainer = pl.Trainer(
        devices=[args.gpu_num],
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, datamodule=dataset)
    print(checkpoint_callback.best_model_path)
    ckpt_path = checkpoint_callback.best_model_path
    trainer.test(model=model, ckpt_path=ckpt_path, datamodule=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='New-Fashion-Product-Performance-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    parser.add_argument('--model_name', type=str, default='Transformer')
    parser.add_argument('--dataset_name', type=str, default='TBH')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=512)
    parser.add_argument('--endo_input_len', type=int, default=52)
    parser.add_argument('--exo_input_len', type=int, default=52)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    # Specific arguments
    parser.add_argument("--use_revin", action='store_true')
    parser.add_argument("--use_endo", action='store_true')
    parser.add_argument("--use_trend", type=bool, default=False)
    parser.add_argument("--use_weather", type=bool, default=False)
    parser.add_argument('--segment_len', type=int, default=4)
    parser.add_argument('--num_exo_vars', type=int, default=3)
    parser.add_argument("--num_meta", type=int, default=52)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='New-Fashion-Product-Performance-Forecasting')
    parser.add_argument('--wandb_dir', type=str, default='./')

    args = parser.parse_args()
    run(args)