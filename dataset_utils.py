from __future__ import print_function, division, absolute_import
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os, sys
import copy
import torch
import torchvision
torchvision.set_image_backend('accimage')
from torchvision import datasets, models, transforms
from  torch.utils.data  import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from collections import OrderedDict
import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from  torch.utils.data import Dataset

organ2id = {
             'adrenal_gland'   :  0,
             'bile_liver'      :  1,
             'bladder'         :  2,
             'brain'           :  3,
             'breast'          :  4,
             'cervix_uteri'    :  5,
             'colon'           :  6,
             'corpus_uteri'    :  7,
             'esophagus'       :  8,
             'kidney'          :  9,
             'liver'           :  1,
             'lung'            : 10,
             'lymphnode'       : 11,
             'ovary'           : 12,
             'pancreas'        : 13,
             'prostate'        : 14,
             'rectosigmoid'    :  6,
             'rectum'          :  6,
             'skin'            : 15,
             'small_intestine' :  6,   
             'stomach'         : 16,
             'testis'          : 17,
             'thymus'          : 18,
             'thyroid'         :  19,
                         
}

id2organ =  {value:key for key, value in organ2id.items()}

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')
  
def default_loader(path):
    return pil_loader(path)

class PTCGA200(Dataset):
    def __init__(self, root_dir, load_in_mem=False, transform=None):
        ext = ['.JPG', '.jpg', '.JPEG', '.png']
        self.load_in_mem = load_in_mem
        self.transform = transform
        self.root_dir = root_dir
        self.dirs = sorted( os.listdir(root_dir) )
        try: 
            self.dirs.remove('thumbnail_position_map')
        except:
            print()
        self.data = []
        self.img_data = [] if self.load_in_mem else None
        self.label = []
        self.data_in_each_label = OrderedDict({str(key): OrderedDict() for key in id2organ.keys()})
        idx = 0
        for d in self.dirs:
            organ_dir = os.path.join(self.root_dir, d)
            if os.path.isdir(organ_dir):
                imgs = sorted( os.listdir(organ_dir) )
                for img in imgs:
                    if img[img.find('.'): ] in ext:
                        self.data.append(os.path.join(self.root_dir, d, img))
                        self.label.append(organ2id[d])
                        self.data_in_each_label[str(organ2id[d])].update({os.path.join(self.root_dir, d, img) : idx })
                        idx += 1
                        
                    if self.load_in_mem:
                        self.img_data.append(default_loader( os.path.join(self.root_dir, d, img)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load_in_mem:
            image = self.img_data[idx]
        else:
            image = default_loader( self.data[idx] )
        return self.transform(image), torch.tensor(self.label[idx], dtype=torch.int64)
 
class PatchCamelyon(Dataset):
    def __init__(self, h5_dir, load_in_mem=False, mode='train', transform=transforms.ToTensor()):
        assert mode in ['train', 'valid', 'test']
        self.load_in_mem = load_in_mem
        self.transform = transform
        self.h5_dir = h5_dir
        self.toImage = transforms.ToPILImage()        
        
        if load_in_mem:
            self.img_data = []
            with h5py.File(os.path.join(h5_dir, '{}_x.h5').format(mode), 'r') as f:
                x = np.zeros(f['x'].shape,dtype=np.uint8)
                f['x'].read_direct(x)
            with h5py.File(os.path.join(h5_dir, '{}_y.h5').format(mode), 'r') as f:
                y = np.zeros(f['y'].shape,dtype=np.uint8)
                f['y'].read_direct(y)
                self.y = torch.tensor(y).view(-1).to(torch.int64)
            for i in range(x.shape[0]):
                self.img_data.append(self.toImage( x[i] ) )
        else:
            self.x = h5py.File(os.path.join(h5_dir, '{}_x.h5'.format(mode)), 'r')
            self.y = h5py.File(os.path.join(h5_dir, '{}_y.h5').format(mode), 'r')
                        
    def __len__(self):
        if self.load_in_mem:
            return( len(self.img_data))
        else:
            return self.y['y'].shape[0]

    def __getitem__(self, idx):
        if self.load_in_mem:
            image = self.img_data[idx]
            return self.transform(image), self.y[idx]
        else:
            image = self.toImage(self.x['x'][idx])
            target =  torch.tensor(self.y['y'][idx][0][0][0], dtype=torch.int64)
            return self.transform(image), target

class PCam200(Dataset):
    def __init__(self, root_dir, mode, load_in_mem=False, transform=None):
        ext = ['.JPG', '.jpg', '.JPEG', '.png']
        self.mode = mode # train, val, test
        self.load_in_mem = load_in_mem
        self.transform = transform
        self.root_dir = root_dir
        self.data = []
        self.img_data = [] if self.load_in_mem else None
        self.label = []
        for nort in os.listdir(os.path.join(root_dir, mode)):
            for case in os.listdir(os.path.join(root_dir, mode, nort)):
                if os.path.isdir(os.path.join(root_dir, mode, nort, case)):
                    imgs = sorted( os.listdir(os.path.join(root_dir, mode, nort, case)) )
                    for img in imgs:
                        if img[img.find('.'): ] in ext:
                            self.data.append(os.path.join(root_dir, mode, nort, case, img))
                            if nort == 'normal':
                                self.label.append(0)
                            elif nort == 'tumor':
                                self.label.append(1)
                            else:
                                continue
                        
                        if self.load_in_mem:
                            self.img_data.append(default_loader( os.path.join(root_dir, mode, nort, case, img)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load_in_mem:
            image = self.img_data[idx]
        else:
            image = default_loader( self.data[idx] )
        return self.transform(image), torch.tensor(self.label[idx], dtype=torch.int64)
    
class segPANDA200(Dataset):
    def __init__(self, root_dir, mode, load_in_mem=False, transform=None):
        ext = ['.JPG', '.jpg', '.JPEG', '.png']
        self.mode = mode # train, val, test
        self.load_in_mem = load_in_mem
        self.transform = transform
        self.root_dir = root_dir
        self.data = []
        self.mask = []
        self.img_data = []
        self.mask_data = []
        self.label = []
        for grade in os.listdir(os.path.join(root_dir, mode)):
            for case in os.listdir(os.path.join(root_dir, mode, grade)):
                if os.path.isdir(os.path.join(root_dir, mode, grade, case)):
                    imgs = sorted( os.listdir(os.path.join(root_dir, mode, grade, case)) )
                    for img in imgs:
                        if img[img.find('.'): ] in ext:
                            if 'mask' in img:
                                continue
                            else:
                                self.data.append( os.path.join(root_dir, mode, grade, case, img) )
                                self.mask.append( os.path.join(root_dir, mode, grade, case,  img[:img.find('.')]+'_mask.png') )
                                if self.load_in_mem:
                                    self.img_data.append(default_loader( self.data[-1]))
                                    self.mask_data.append(default_loader( self.mask[-1]))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load_in_mem:
            image = self.img_data[idx]
            mask = self.mask_data[idx]
        else:
            image = default_loader( self.data[idx] )
            mask = default_loader( self.mask[idx] ).convert('RGB')
        return self.transform(image), torch.from_numpy(np.array(mask)[:, :, 0])
    

def prepare_dataloaders(config, tra, tra_val):
    if 'PTCGA' in os.path.basename(config["dataset_dir"]):
        dataset_t = PTCGA200(config.dataset_dir, transform=tra)
        dataset_v = copy.deepcopy(dataset_t)
        dataset_v.transform = tra_val
        train_idx = config.fold_dict[str(0)]['train']
        val_idx   = config.fold_dict[str(0)]['val']
        test_idx   = config.fold_dict[str(0)]['test']
        train_dataset = Subset(dataset_t, train_idx)
        val_dataset   = Subset(dataset_v, val_idx)
        test_dataset   = Subset(dataset_v, test_idx)

        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_sampler_v = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler     = torch.utils.data.distributed.DistributedSampler(test_dataset)

    elif 'segPANDA' in  os.path.basename(config["dataset_dir"]):
        train_dataset   = segPANDA200(config.dataset_dir, mode='train', transform=tra)
        val_dataset     = segPANDA200(config.dataset_dir, mode='val', transform=tra_val)
        test_dataset    = segPANDA200(config.dataset_dir, mode='test', transform=tra_val)
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler     = torch.utils.data.SequentialSampler(test_dataset)

    elif 'PCam200' in  os.path.basename(config["dataset_dir"]):
        train_dataset = PCam200(config.dataset_dir, mode='train', transform=tra)
        val_dataset   = PCam200(config.dataset_dir, mode='val', transform=tra_val)
        test_dataset   = PCam200(config.dataset_dir, mode='test', transform=tra_val)
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_sampler_v = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler     = torch.utils.data.distributed.DistributedSampler(test_dataset)
        

    elif 'PCam' in  os.path.basename(config["dataset_dir"]):
        train_dataset = PatchCamelyon(config.dataset_dir, load_in_mem=True, transform=tra, mode='train')
        val_dataset = PatchCamelyon(config.dataset_dir, load_in_mem=True, transform=tra_val, mode='valid')
        test_dataset  = PatchCamelyon(config.dataset_dir, load_in_mem=True, transform=tra_val, mode='test')
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_sampler_v = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler     = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        raise(NotImplementedError)
    
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler, num_workers=config.num_workers, drop_last=True) 
    dataloader_v = DataLoader(val_dataset, batch_size=config.batch_size*2, shuffle=(val_sampler is None), pin_memory=True, sampler=val_sampler, num_workers=config.num_workers)
    dataloader_train_v = DataLoader(train_dataset, batch_size=config.batch_size*2, shuffle=(train_sampler_v is None), sampler=train_sampler_v, pin_memory=True, num_workers=config.num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=config.batch_size*2, shuffle=(test_sampler is None), pin_memory=True, sampler=test_sampler, num_workers=config.num_workers)
    return train_sampler, val_sampler, train_sampler_v, dataloader, dataloader_v, dataloader_train_v, dataloader_test