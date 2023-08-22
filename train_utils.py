import os, re
import time
import shutil
import tqdm
import torch
import torch.nn as nn
import timm
from logger import  DummyWriter, Logger
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import dataset_utils
import torchvision.transforms as transforms
import kornia
import utils
import byol_torch

def get_next_run_id(results_dir):
    idx = []
    p = r'^[0-9]+'
    for dirname in os.listdir(results_dir):
        m = re.match(p, dirname)
        if m is not None and os.path.isdir(os.path.join(results_dir, dirname)):
            idx.append( int(m.group()) )
    if len(idx) == 0:
        run_id = 0
    else:
        run_id = max(idx)
    return run_id + 1

def create_project_dir(results_dir, project_name):
    run_id = get_next_run_id(results_dir)
    project_dir = os.path.join(results_dir, '{:06}'.format(run_id) + '-' + project_name)
    os.makedirs(project_dir , exist_ok=True)
    return project_dir

def setup_project(config):
    if config.rank==0:
        timestamp = time.strftime("%b%d%Y_%H%M%SUTC", time.gmtime())
        config['timestamp'] = timestamp
        project_dir = create_project_dir('./runs', str(config.memo) + str(timestamp))   
        log_dir = os.path.join(project_dir, 'tf_log')
        os.makedirs(log_dir, exist_ok=True)
        snapshot_dir = os.path.join(project_dir, 'snapshot')
        os.makedirs(snapshot_dir, exist_ok=True)
        config["project_dir"] = project_dir
        config["log_dir"] = log_dir 
        config['snapshot_dir'] = snapshot_dir
        if '__file__' in locals():
            print('Executing ...:    ', __file__)
            for f in os.listdir('./'):
                if '.py' in f:
                    shutil.copy(f, os.path.join(project_dir, f))

    return config

def setup_config(config):
    if 'PTCGA' in os.path.basename(config["dataset_dir"]):
        if os.path.isfile(os.path.join(config.root_dir, '3fold_dict_idx_filenames.pickle')):
            import pickle
            with open(os.path.join(config.root_dir, '3fold_dict_idx_filenames.pickle'), 'rb') as f:
                fold_dict = pickle.load(f)
        else:
            dataset_t = dataset_utils.PTCGA200(config.dataset_dir, transform=None)
            print("Dataset size: ", len(dataset_t))
            fold_dict = utils.sample_CV(dataset_t, n_trial=config.n_fold)
        print("Train size: ", len(fold_dict['0']['train']['idx']))
        print("Validation size: ", len(fold_dict['0']['val']['idx']))
        print("Test size: ", len(fold_dict['0']['test']['idx']))
    
        idx_dict = {}
        for nth_fold in range(config.n_fold):
            idx_dict.update({str(nth_fold):
                                            {'train': fold_dict[str(nth_fold)]['train']['idx'],
                                            'val': fold_dict[str(nth_fold)]['val']['idx'],
                                            'test': fold_dict[str(nth_fold)]['test']['idx']}
                            })
    else:
        idx_dict = {}
    config['fold_dict'] = idx_dict
    return config

def setup_logger(config):
    if config.rank==0:
        txtlogger = Logger(file_name=os.path.join(config['project_dir'], "log.txt"), file_mode="w", should_flush=True)
        return SummaryWriter(config['log_dir']), txtlogger
    else:
        return DummyWriter(''), None

def init_model(config):
    if config.self_supervised==None:
        if config.model_name in ["resnet18", "resnet50"]:
            model = torch.hub.load('pytorch/vision:v0.6.0', config.model_name, 
                            pretrained=config['pretrained'])
            model.fc = model.fc = nn.Sequential(
            nn.Dropout( config.dropout_p ),
            nn.Linear(in_features=model.fc.in_features, out_features=config.num_classes, bias=True)
            )
            if config['fine_tune']:
                for name, param in model.named_parameters():
                    if 'fc' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if 'segPANDA' in  os.path.basename(config["dataset_dir"]):
                model = utils.FCN_CNN_resnet(model, config['num_classes'])
        
        elif config.model_name == "incv3":
            model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', 
                            pretrained=config['pretrained'])
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, config['num_classes'], bias=True)
            model.fc = nn.Linear(model.fc.in_features, config['num_classes'], bias=True)
            model.dropout.p = config.dropout_p
            if config['fine_tune']:
                for name, param in model.named_parameters():
                    if 'fc' in name:
                        param.requires_grad = True
                    elif 'classifier.6' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if 'segPANDA' in  os.path.basename(config["dataset_dir"]):
                model = utils.FCN_CNN_incv3(model, config['num_classes'])
        
        elif config.model_name == "eff-b3":
            # Caveat: the original study used https://github.com/lukemelas/EfficientNet-PyTorch
            model = torchvision.models.efficientnet_b3(pretrained=config['pretrained'])
            model.classifier[0].p = config.dropout_p
            model.classifier[1] = nn.Linear(model.classifier.fc.in_features, config['num_classes'], bias=True)
            if config['fine_tune']:
                for name, param in model.named_parameters():
                    if 'classifier' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if 'segPANDA' in  os.path.basename(config["dataset_dir"]):
                model = utils.FCN_CNN_eff(model, config['num_classes'])
        
        elif 'vit' in config.model_name:
            if config.model_name == "vit-s16":
                patch_size = 16 
                model = timm.create_model('vit_small_patch16_224_in21k', num_classes=config['num_classes'])
            elif config.model_name == "vit-b32":
                patch_size = 32
                model = timm.create_model('vit_base_patch32_224_in21k', num_classes=config['num_classes'])
            pos_embed_target = torch.zeros(1, 1+int(config.image_size/patch_size)**2, model.num_features)
            model.pos_embed = nn.Parameter( timm.models.vision_transformer.resize_pos_embed(model.pos_embed, pos_embed_target) )
            model.patch_embed.img_size = (config.image_size, config.image_size)
            if 'segPANDA' in  os.path.basename(config["dataset_dir"]):
                model = utils.FCN_CNN_vit(model, config['num_classes'])
    else:
        if config.model_name =="resnet18":
            encoder_online = byol_torch.ResNet18(initializer='original')
            encoder_target = byol_torch.ResNet18(initializer='original')

        elif config.model_name =="resnet50":
            encoder_online = byol_torch.ResNet50(initializer='original')
            encoder_target = byol_torch.ResNet50(initializer='original')   
        if config.self_supervised=='byol':
            model = byol_torch.BYOL(encoder_online=encoder_online, encoder_target=encoder_target, **config.model_config)
    
    if config['reset_bn']:
        for n, m in model.named_modules():
            if 'bn' in n:
                m.reset_running_stats()
                m.reset_parameters()
    nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if config['checkpoint'] is not None:
        model = torch.load(config['checkpoint'], strict=False)
    return model
    
def save_ckpt(config, model, optimizer, scheduler, scaler, epoch, itr, num_updates):
    snapshot_path = os.path.join(config.snapshot_dir, f'{config.memo}_{epoch}epochs_{num_updates}.pth.tar')
    ckpt = {}
    ckpt.update(
{
    'state_dict':     model.module.state_dict(),
    'opt_state_dict': optimizer.state_dict(),
    'sch_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'epoch': epoch,
    'itr'  : itr,
    'num_updates': num_updates,
})
    torch.save(ckpt, snapshot_path)


def prepare_aug(config):
    if 'PTCGA' in os.path.basename(config["dataset_dir"]):
        if config['self_supervised'] is not None:
            tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5), 
                                transforms.ToTensor(),
                                transforms.Normalize(*config.image_stats)  ])
        else:
            tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                #transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5), 
                                transforms.ToTensor(),
                                transforms.Normalize(*config.image_stats)  ])

        tra_val = transforms.Compose([  transforms.CenterCrop(287),                # equivalent to 20x (0.5mpp)
                                        transforms.Resize(config['val_image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stats)  ])
        tra_kornia = None

    elif 'segPANDA' in  os.path.basename(config["dataset_dir"]):
        tra = transforms.Compose([     
                                   transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                   transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                   transforms.ToTensor(),
                                   transforms.Normalize(*config.image_stats) ])

        tra_val = transforms.Compose([  
                                        transforms.CenterCrop(config['val_image_size']), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stats) ])
        
        tra_kornia = [
                        kornia.augmentation.RandomCrop((config['image_size'], config['image_size'])),
                        kornia.augmentation.RandomHorizontalFlip(p=0.5),
                        kornia.augmentation.RandomVerticalFlip(p=0.5),
                    ]

    elif 'PCam200' in  os.path.basename(config["dataset_dir"]):
        tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                   transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                   #transforms.RandomGrayscale(p=0.2),
                                   transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5), 
                                   transforms.ToTensor(),
                                   transforms.Normalize(*config.image_stats)  ])

        tra_val = transforms.Compose([  transforms.CenterCrop(287),                # equivalent to 20x (0.5mpp)
                                        transforms.Resize(config['val_image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stats)  ])
        tra_kornia = None

    elif 'PCam' in  os.path.basename(config["dataset_dir"]):
        tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                   transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                   #transforms.RandomGrayscale(p=0.2),
                                   transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5), 
                                   transforms.ToTensor(),
                                   transforms.Normalize(*config.image_stats)  ])

        tra_val = transforms.Compose([     #transforms.CenterCrop(287),                # equivalent to 20x (0.5mpp)
                                        transforms.Resize(config['val_image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stats)  ])
        tra_kornia = None

    else:
        raise(NotImplementedError)

    return tra, tra_val, tra_kornia

@torch.no_grad()
def Accuracy(config, model, criterion, dataloader, mode, num_itr=None):
    if num_itr is None:
        num_itr = len(dataloader)
    correct = torch.tensor([0]).to(config.gpu)
    total = torch.tensor([0]).to(config.gpu)
    val_loss = torch.tensor([0.0])
    loss_flag = 0
    for j, batch_v in enumerate( dataloader ):
        model.eval()
        val_in     = batch_v[0].to(config.gpu, non_blocking=True)
        val_target = batch_v[1].to(config.gpu, torch.int64, non_blocking=True)
        with autocast(enabled=config['enable_autocast']): 
            output  = model(val_in)       
            if not loss_flag:
                val_loss = criterion(output, val_target) / float( config["num_accum"] )
                loss_flag = 1
        total += val_target.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == val_target).sum().item()
        if j > num_itr:
            break
    torch.distributed.all_reduce(total) 
    torch.distributed.all_reduce(correct) 
    
    acc = correct / total
    if config.rank==0:
        print ("{} accuracy: ".format(mode), 100.0 * acc)
        print ("{} loss: ".format(mode), val_loss)
    return acc

@torch.no_grad()
def DiceJaccardAccuracy(model, loader, device, ddp=False, ignor_index=[], num_classes=6, num_img=None, eps=1e-5):
    """
    return:
        micro mean IoU,
        micro mean Dice,
        macro mean IoU,
        macro mean Dice,
        mean Pixel Accuracy
    """
    model.eval()
    if ddp:
        TPs = torch.tensor([0] * num_classes).to(device, torch.int64)
        FPs = torch.tensor([0] * num_classes).to(device, torch.int64)
        FNs = torch.tensor([0] * num_classes).to(device, torch.int64)
    else:
        TPs = [0] * num_classes
        FPs = [0] * num_classes
        FNs = [0] * num_classes
    img_loaded = 0
    for i, batch in enumerate(loader):
        img_loaded += len( batch )
        img = batch[0].to(device, non_blocking=True)
        mask = batch[1].to(device, torch.uint8, non_blocking=True)
        with autocast():
            out = model(img)['out']
        pred = out.argmax(1).to(torch.uint8)
        
        
        for label in range(num_classes):
            if not label in ignor_index:
                TPs[label] += int((torch.where(mask==label, 1, 0) * torch.where(pred==label, 1, 0)).sum())
                FPs[label] += int((torch.where(mask!=label, 1, 0) * torch.where(pred==label, 1, 0)).sum())
                FNs[label] += int((torch.where(mask==label, 1, 0) * torch.where(pred!=label, 1, 0)).sum())
        if num_img is not None and img_loaded > num_img:
            break
    if ddp:
        torch.distributed.all_reduce(TPs) 
        torch.distributed.all_reduce(FPs)
        torch.distributed.all_reduce(FNs) 
    ious  = []
    dices = []
    pas   = []
    for label in range(num_classes):
        ious  += [ float(TPs[label])/(TPs[label] + FPs[label] + FNs[label] + eps) ]
        dices += [ 2*float(TPs[label])/(2*TPs[label] + FPs[label] + FNs[label] + eps) ]
        pas   += [ float(TPs[label])/(TPs[label] + FNs[label] + eps) ]
    if ddp:
        return (float(sum(TPs))/(sum(TPs) + sum(FPs) + sum(FNs) + eps)).item(), (2*float(sum(TPs))/(2*sum(TPs) + sum(FPs) + sum(FNs) + eps)).item(), (sum(ious)/num_classes).item(), (sum(dices)/num_classes).item(), (sum(pas)/num_classes).item()
    else:
        return (float(sum(TPs))/(sum(TPs) + sum(FPs) + sum(FNs) + eps)), (2*float(sum(TPs))/(2*sum(TPs) + sum(FPs) + sum(FNs) + eps)), (sum(ious)/num_classes), (sum(dices)/num_classes), (sum(pas)/num_classes)