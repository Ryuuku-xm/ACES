import argparse
import os
from datetime import datetime

import numpy as np
import torch

from config_files.parameter import Config
from dataloader import data_generator
from model import HDSM
from trainer import Trainer
from utils.logger import _logger


def set_seed(seed):
    SEED = seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    return seed


def main(args, configs, seed=None):
    method = "HDSM"
    # load dataset
    dataset = args.task
    data_path = f"./dataset/{dataset}"  # './dataset/Mondrian'
    train_dl, val_dl, test_dl = data_generator(data_path, configs)

    # set seed
    if seed is not None:
        seed = set_seed(seed)
    else:
        seed = set_seed(args.seed)

    # set log
    experiment_log_dir = os.path.join("./experiments_logs", f"seed_{seed}_lr_{configs.lr}_epoch_{configs.epoch}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Seed: {seed}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Learning rate:    {configs.lr}')
    logger.debug(f'Epochs:    {configs.epoch}')
    logger.debug(f'Temperature: {configs.temperature}')
    logger.debug("=" * 45)

    # load model
    model = HDSM(configs, args).to(device)
    params_group = [{'params': model.parameters()}]
    model_optimizer = torch.optim.Adam(params_group, lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=1e-4)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.epoch)

    # Trainer
    trainer = Trainer(model, model_optimizer, model_scheduler, train_dl, val_dl, test_dl, device, logger,
                      args, configs, experiment_log_dir, seed)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="Mondrian", type=str, help="select task")
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--seed', default=2023, type=int, help='seed value')
    parser.add_argument('--embed', default='coopeformer', type=str, help='the way of embedding')

    args = parser.parse_args()
    device = torch.device(args.device)
    configs = Config()
    main(args, configs)
