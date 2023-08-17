# coding: utf-8
from __future__ import print_function, division, absolute_import
import os
import datetime.timedelta as timedelta
import socket
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision
torchvision.set_image_backend('accimage')
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import train_utils
import dataset_utils
import utils
from config import config

def main_worker(gpu, ngpus_per_node, config):
    train_start = time.time()
    batch_size = config["batch_size"]
    epochs     = config["epochs"]
    root_dir = config["root_dir"]
    dataset_dir = config["dataset_dir"]
    utils.seed_rng(config.seed)
    config.gpu = gpu
    config.rank = ngpus_per_node*config.hosts.index(socket.gethostname()) + gpu
    print ('GPU ', gpu, 'Rank ', config.rank, "World size ", config.world_size, "dist_url ", config.dist_url)
    dist.init_process_group(backend='nccl', init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank) 
    train_utils.setup_project(config)
    writer, txtlogger = train_utils.setup_logger(config)
    print ("Setup logger in {}".format(config.gpu))
    torch.backends.cudnn.benchmark = True
    if config.checkpoint is not None:
        ckpt = torch.load(os.path.join(config.root_dir, config.checkpoint), map_location=torch.device('cpu'))
        resume_epoch = ckpt['epoch']
    else:
        resume_epoch = 0
    print("Train preprocessing time: ", str(timedelta(seconds=time.time() - train_start)) )    

    model = train_utils.init_model(config)
    utils.print_num_parameters(model)
    if config.rank==0:
        for k, v in config.items():
            if "fold_dict" not in k:
                print (k, " : ", v)                    
        print ("Project dir: ", config['project_dir'])
    if config.checkpoint is not None:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(config.gpu)
    model = DDP(model, device_ids=[config.gpu], output_device=config.gpu)   
    criterion = nn.CrossEntropyLoss()
    
    tra, tra_val, tra_kornia = train_utils.prepare_aug(config)
    train_sampler, val_sampler, train_sampler_v, dataloader, dataloader_v, dataloader_train_v, dataloader_test = dataset_utils.prepare_dataloaders(config, tra, tra_val)   

    lr = config.lr * (batch_size*config.world_size*config["num_accum"] // config["base_batch_size"]) 
    cold_step = len(dataloader) * config['cold_epochs'] //config["num_accum"]
    peak_step = cold_step + len(dataloader) * config['warmup_epochs']//config["num_accum"]
    max_step = len(dataloader) * epochs//config["num_accum"]

    wd_params = []
    no_wd_params = []
    for name, param in  model.named_parameters():
        if 'bn' in name:
            no_wd_params.append(param)
        elif name in ['cls_token', 'pos_embed']:
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    if config.optin == 'sgd':
        optimizer = optim.SGD([{'params': no_wd_params, 'weight_decay': 0.0}, {'params': wd_params}], lr=lr, 
                                momentum=0.9, eps=1e-08, weight_decay=config["wd"])
    elif config.optin == 'adamw':
        optimizer = optim.AdamW([{'params': no_wd_params, 'weight_decay': 0.0}, {'params': wd_params}], lr=lr, 
                                betas=(0.9, 0.999), eps=1e-08, weight_decay=config["wd"])
    elif config.optin == 'adam':
        optimizer = optim.Adam([{'params': no_wd_params, 'weight_decay': 0.0}, {'params': wd_params}], lr=lr, 
                                betas=(0.9, 0.999), eps=1e-08, weight_decay=config["wd"])
    elif config.optin == 'lars':
        optimizer = optim.SGD([{'params': no_wd_params, 'weight_decay': 0.0, 'layer_adaptation': False},
                           {'params': wd_params, 'layer_adaptation': True}], lr=lr, momentum=0.9, weight_decay=config["wd"])
        optimizer = utils.LARS(optimizer, 0.001)
    optimizer.zero_grad()  
    scheduler = utils.GradualWarmupWithCosineLR(optimizer, cold_step, peak_step, max_step, initial_lr=config.lr) 
    scaler = GradScaler()    # for mixed prescion training
    if config.checkpoint is not None:
        itr = ckpt['itr']
        num_updates = ckpt['num_updates']  
        for _ in range(num_updates):
            scheduler.step()  
        optimizer.load_state_dict(ckpt['opt_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    else:
        itr = 0
        num_updates = 0   

    for epoch in range(resume_epoch+1, epochs+1):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_sampler_v.set_epoch(epoch)
        
        # validate every epoch 
        if config.rank==0:
            print (f' {num_updates}th updates, {epoch}/{epochs} epoch')
        val_acc, val_loss = train_utils.Accuracy(config, model, criterion, dataloader_v, 'Val', len(dataloader_v) )
        writer.add_scalar('Misc/val_acc', 100.0 * val_acc.item(), num_updates)
        writer.add_scalar('Loss/val', val_loss.item(), num_updates)
        
        train_v_acc, train_v_loss = train_utils.Accuracy(config, model, criterion, dataloader_train_v, 'Train', len(dataloader_v) )
        writer.add_scalar('Misc/train_acc', 100.0 * train_v_acc.item(), num_updates)
        
        pbar = tqdm(dataloader)
        for batch in pbar:
            itr += 1
            if tra_kornia is not None:
                data = batch[0].to(config.gpu, non_blocking=True)
                target = batch[1].to(torch.float32).to(config.gpu, non_blocking=True)

                for aug in tra_kornia:
                    data = aug(data)
                
                target = target.unsqueeze(1).repeat(1, 3, 1, 1) 
                for aug in tra_kornia:
                    target = aug(target, params=aug._params)
                target = target[:, 0, :, :].to(torch.int64)
            else:
                data = batch[0].to(config.gpu, non_blocking=True)
                target = batch[1].to(torch.int64).to(config.gpu, non_blocking=True)

            model.train()
            with autocast(enabled=config['enable_autocast']): 
                if 'segPANDA' in  os.path.basename(config["dataset_dir"]):
                    output = model(data)['out']
                else:
                    output = model(data)
                loss = criterion(output, target)
                loss = loss / float( config["num_accum"] )

            scaler.scale(loss).backward()
            if (itr+1)%config["num_accum"] == 0:
                if config["clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),  config["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                num_updates += 1
                if (num_updates)%5==0 and config.rank==0:
                    print (f"Train loss @rank{config.rank}: {loss.item()}")
                    writer.add_scalar('Loss/train', loss.item(), num_updates)
        
            if (num_updates)%100==0 and config.rank==0:
                print ('{}th updates, {}/{} epoch:'.format(num_updates, epoch, epochs))
                writer.add_scalar('Misc/itr', itr, num_updates)
                writer.add_scalar('Misc/max_memory_allocated', torch.cuda.max_memory_allocated(device=config.gpu) / 2**30, num_updates)
                writer.add_scalar('Misc/max_memory_cached', torch.cuda.max_memory_reserved(device=config.gpu) / 2**30, num_updates)
                for param_group in optimizer.param_groups:
                    print('lr: ', param_group['lr'])
                    writer.add_scalar('Misc/lr', param_group['lr'], num_updates)
            if 'segPANDA' in os.path.basename(config["dataset_dir"]) and (num_updates)%200==0:
                results = train_utils.DiceJaccardAccuracy(model, dataloader_v, writer, config.gpu, ddp=True) 
                for tag, value in zip(['mIoU_micro', 'mDice_micro', 'mIoU_macro', 'mDice_macro', 'mPA'], results):
                    writer.add_scalar(f'Misc/val_{tag}', value, num_updates)
            elif 'PCam' in os.path.basename(config["dataset_dir"]) and (num_updates)%200==0:
                val_acc = train_utils.Accuracy(config, model, criterion, dataloader_v, 'Val', len(dataloader_v) )
                writer.add_scalar('Misc/val_acc', 100.0 * val_acc.item(), num_updates)
        if config.rank==0:            
            print("{}th epoch time: ".format(epoch), str(timedelta(seconds=time.time() - epoch_start)) )
            print("Elapsed time: ", str(timedelta(seconds=time.time() - train_start)) )
        if (epoch)%config['snapshot_epoch']==0 and config.rank==0:
            train_utils.save_ckpt(config, model, optimizer, scheduler, scaler, epoch, itr, num_updates)
        writer.add_scalar('Misc/num_epochs', float(epoch), num_updates)
    # testing and epilogue 
    print("Elapsed time: ", str(timedelta(seconds=time.time() - train_start)) )
    if 'segPANDA' in os.path.basename(config["dataset_dir"]):
        if config.rank==0:
            results = train_utils.DiceJaccardAccuracy(model, dataloader_test, config.gpu, ddp=False, num_img=None)
            for tag, value in zip(['mIoU_micro', 'mDice_micro', 'mIoU_macro', 'mDice_macro', 'mPA'], results):
                writer.add_scalar(f'Misc/test_{tag}', value, num_updates)
            print( results)
    else :
        test_acc = train_utils.Accuracy(config, model, criterion, dataloader_test, 'Test')
        if config.rank==0:
            print("Test accuracy: ", test_acc)
            writer.add_scalar('Misc/test_acc', 100.0 * test_acc.item(), num_updates)
    if config.rank==0:
        train_utils.save_ckpt(config, model, optimizer, scheduler, scaler, epoch, itr, num_updates)
    print ("Finished !")
    print("Elapsed time: ", str(timedelta(seconds=time.time() - train_start)) )
    txtlogger.close()
        
# DDP setup
def main(config):
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node * config.nnode
    config = train_utils.setup_config(config)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

if __name__ == '__main__':
    main(config)