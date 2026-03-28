import numpy as np
import os, sys
from tqdm import tqdm
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='./options/test/LOLBlur.yml', help = 'Config file of testing')
args = parser.parse_args()

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need
path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0" # you need to fix this before importing torch

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models
from utils.test_utils import *
from ptflops import get_model_complexity_info

#parameters for saving model
PATH_MODEL= create_path_models(opt['save'])


def load_model(model, path_weights):

    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    # print(checkpoints.keys())
    # sys.exit()
    weights = checkpoints['params']

    model_keys = list(model.state_dict().keys())
    model_uses_module = len(model_keys) > 0 and model_keys[0].startswith('module.')
    weight_keys = list(weights.keys())
    weights_use_module = len(weight_keys) > 0 and weight_keys[0].startswith('module.')

    if model_uses_module and not weights_use_module:
        weights = {'module.' + key: value for key, value in weights.items()}
    elif not model_uses_module and weights_use_module:
        weights = {key.replace('module.', '', 1): value for key, value in weights.items()}

    model.load_state_dict(weights)
    print('Loaded weights correctly')
    
    return model

def run_evaluation(rank, world_size):
    use_distributed = world_size > 1
    if use_distributed:
        # NCCL is not available on native Windows. Use GLOO there.
        backend = 'nccl' if torch.cuda.is_available() and os.name != 'nt' else 'gloo'
        setup(rank, world_size=world_size, backend=backend)
    # LOAD THE DATALOADERS
    test_loader, _ = create_test_data(rank, world_size=world_size, opt = opt['datasets'])
    # DEFINE NETWORK
    model, _, _ = create_model(opt['network'], rank=rank)

    model = load_model(model, opt['save']['path'])
    metrics_eval = {}
    print("Model loaded. Calling eval_model...")

    if use_distributed:
        dist.barrier()
    # eval phase
    model.eval()
    print("model.eval() set. Proceeding to eval_model...")
    metrics_eval, _ = eval_model(model, test_loader, metrics_eval, rank=rank, world_size=world_size, eta=False, eval_lpips=False)
    print("eval_model completed.")
    if use_distributed:
        dist.barrier()
    # print some results
    if rank==0:
        if type(next(iter(metrics_eval.values()))) == dict:
            for key, metric_eval in metrics_eval.items():
                print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
        else:
            print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
            with open('run_metrics.txt', 'w') as f:
                f.write(f"dataset={opt['datasets']['name']}\n")
                f.write(f"PSNR={metrics_eval['valid_psnr']}\n")
                f.write(f"SSIM={metrics_eval['valid_ssim']}\n")
                f.write(f"LPIPS={metrics_eval['valid_lpips']}\n")
    if use_distributed:
        cleanup()

def main():
    world_size = 1
    if world_size > 1:
        mp.spawn(run_evaluation, args =(world_size,), nprocs=world_size, join=True)
    else:
        run_evaluation(rank=0, world_size=world_size)

if __name__ == '__main__':
    main()
