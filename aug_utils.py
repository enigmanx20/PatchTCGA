import os
import torchvision.transforms as transforms
import kornia
import utils

def prepare_aug(config):
    if 'PTCGA' in os.path.basename(config["dataset_dir"]):
        if config['self_superversed'] is not None:
            tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5), 
                                transforms.ToTensor(),
                                transforms.Normalize(*config.image_stat)  ])
            
            tra = utils.ApplyTwoTransforms(tra, tra)
        else:
            tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                #transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5), 
                                transforms.ToTensor(),
                                transforms.Normalize(*config.image_stat)  ])

        tra_val = transforms.Compose([  transforms.CenterCrop(287),                # equivalent to 20x (0.5mpp)
                                        transforms.Resize(config['vaal_image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stat)  ])

    elif 'segPANDA' in  os.path.basename(config["dataset_dir"]):
        tra = transforms.Compose([     
                                   transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                   transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                   transforms.ToTensor(),
                                   transforms.Normalize(*config.image_stat) ])

        tra_val = transforms.Compose([  
                                        transforms.CenterCrop(config['val_image_size']), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stat) ])
        
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
                                   transforms.Normalize(*config.image_stat)  ])

        tra_val = transforms.Compose([  transforms.CenterCrop(287),                # equivalent to 20x (0.5mpp)
                                        transforms.Resize(config['vaal_image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stat)  ])

    elif 'PCam' in  os.path.basename(config["dataset_dir"]):
        tra = transforms.Compose([ transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.), ratio=(0.75, 1.3333333333333333)),
                                   transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8), # cf https://arxiv.org/pdf/1703.02442.pdf
                                   #transforms.RandomGrayscale(p=0.2),
                                   transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5), 
                                   transforms.ToTensor(),
                                   transforms.Normalize(*config.image_stat)  ])

        tra_val = transforms.Compose([     #transforms.CenterCrop(287),                # equivalent to 20x (0.5mpp)
                                        transforms.Resize(config['vaal_image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*config.image_stat)  ])
        tra_kornia = None

    else:
        raise(NotImplementedError)

    return tra, tra_val, tra_kornia