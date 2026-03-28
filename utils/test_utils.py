import torch
import torch.distributed as dist
import sys, os
from lpips import LPIPS
import numpy as np
sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses import *
from tqdm import tqdm

calc_SSIM = SSIM(data_range=1.)

def _get_device(rank):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}')
    return torch.device('cpu')

#---------- Set of functions to work with DDP
def setup(rank, world_size, Master_port = '12355', backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = Master_port
    if backend == 'gloo':
        os.environ.setdefault('USE_LIBUV', '0')
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def reduce_tensor(tensor, world_size):
    if not dist.is_available() or not dist.is_initialized() or world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def save_model(model, path):
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), path)

def shuffle_sampler(samplers, epoch):
    '''
    A function that shuffles all the Distributed samplers in the loaders.
    '''
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)

def eval_one_loader(model, test_loader, metrics, rank=0, world_size = 1, eta = False, eval_lpips=True):
    device = _get_device(rank)
    calc_LPIPS = LPIPS(net = 'vgg', verbose=False).to(device) if eval_lpips else None
    mean_metrics = {'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[]}

    if eta: pbar = tqdm(total = int(len(test_loader)))
    print(f"Starting eval over {len(test_loader)} batches...")
    import sys; sys.stdout.flush()
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for idx, (high_batch_valid, low_batch_valid) in enumerate(test_loader):
            print(f"Batch {idx} loaded")
            sys.stdout.flush()
            high_batch_valid = high_batch_valid.to(device)
            low_batch_valid = low_batch_valid.to(device)

            print(f"Running model on batch {idx}...")
            sys.stdout.flush()
            enhanced_batch_valid = model(low_batch_valid)
            print(f"Model inference done for batch {idx}...")
            sys.stdout.flush()
            # loss
            valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
            valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
            if eval_lpips:
                valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())
            else:
                mean_metrics['valid_lpips'].append(0.0)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
    
            mean_metrics['valid_psnr'].append(valid_psnr_batch.item())
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())

            if eta: pbar.update(1)

    valid_psnr_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_psnr'])).to(device), world_size=world_size)
    valid_ssim_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_ssim'])).to(device),world_size=world_size)
    valid_lpips_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_lpips'])).to(device), world_size=world_size)

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()
    
    
    imgs_dict = {'input':low_batch_valid[0], 'output':enhanced_batch_valid[0], 'gt':high_batch_valid[0]}
    
    if eta: pbar.close()
    return metrics, imgs_dict

def eval_model(model, test_loader, metrics, rank=None, world_size = 1, eta = False, eval_lpips=True):
    '''
    This function runs over the multiple test loaders and returns the whole metrics.
    '''
    #first you need to assert that test_loader is a dictionary
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}
    if len(test_loader) > 1:
        all_metrics = {}
        all_imgs_dict = {}
        for key, loader in test_loader.items():

            all_metrics[f'{key}'] = {}
            metrics, imgs_dict = eval_one_loader(model, loader['loader'], all_metrics[f'{key}'], rank=rank, world_size=world_size, eta=eta, eval_lpips=eval_lpips)
            all_metrics[f'{key}'] = metrics
            all_imgs_dict[f'{key}'] = imgs_dict
        return all_metrics, all_imgs_dict
    
    else:
        metrics, imgs_dict = eval_one_loader(model, test_loader['data'], metrics, rank=rank, world_size=world_size, eta=eta, eval_lpips=eval_lpips)
        return metrics, imgs_dict

def eval_one_loader_two_models(model1, model2, test_loader, metrics, devices = ['cuda:0', 'cuda:1'], eta = False):
    calc_LPIPS = LPIPS(net = 'vgg', verbose=False).to(devices[0])
    mean_metrics = {'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[]}

    if eta: pbar = tqdm(total = int(len(test_loader)))
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid, low_batch_valid in test_loader:

            high_batch_valid = high_batch_valid.to(devices[0])
            low_batch_valid = low_batch_valid.to(devices[0])         

            enhanced_batch_valid = model1(low_batch_valid)
            enhanced_batch_valid = torch.clamp(enhanced_batch_valid, 0., 1.)
            enhanced_batch_valid = model2(enhanced_batch_valid.to(devices[1]))
            # loss
            enhanced_batch_valid = enhanced_batch_valid.to(devices[0])
            valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
            valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
            valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
            # print(valid_loss_batch)
            mean_metrics['valid_psnr'].append(valid_psnr_batch.item())
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())
            mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())
            # print(valid_psnr_batch.item())
            if eta: pbar.update(1)
    print(mean_metrics['valid_psnr'])
    valid_psnr_tensor = np.mean(mean_metrics['valid_psnr'])
    valid_ssim_tensor = np.mean(mean_metrics['valid_ssim'])
    valid_lpips_tensor = np.mean(mean_metrics['valid_lpips'])

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()
    
    
    imgs_dict = {'input':low_batch_valid[0], 'output':enhanced_batch_valid[0], 'gt':high_batch_valid[0]}
    
    if eta: pbar.close()
    return metrics, imgs_dict

def eval_model_two_models(model1, model2, test_loader, metrics, devices=['cuda:0', 'cuda:1'], eta = False):
    '''
    This function runs over the multiple test loaders and returns the whole metrics.
    '''
    #first you need to assert that test_loader is a dictionary
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}
    if len(test_loader) > 1:
        all_metrics = {}
        all_imgs_dict = {}
        for key, loader in test_loader.items():

            all_metrics[f'{key}'] = {}
            metrics, imgs_dict = eval_one_loader_two_models(model1, model2, loader['loader'], all_metrics[f'{key}'], devices = devices, eta=eta)
            all_metrics[f'{key}'] = metrics
            all_imgs_dict[f'{key}'] = imgs_dict
        return all_metrics, all_imgs_dict
    
    else:
        metrics, imgs_dict = eval_one_loader_two_models(model1, model2, test_loader['data'], metrics, devices = devices, eta=eta)
        return metrics, imgs_dict