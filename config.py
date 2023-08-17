import easydict
import os

config = {  
                'memo'   :  'resnet50_PTCGA200_scratch',
      'model_name'       :  'resnet50', # one of resnet18, resnet50, incv3, eff-b3, vit-s16, vit-b32
      'dropout_p'        :        0.4,
      'reset_bn'         :       True,
      'self_superversed' :       None, # one of None,byol. None indicates supervised training. BYOL only supports resnet50 now.  
      'fine_tune'        :      False, # True for fine-tuning, False for training from scratch
      'enable_autocast'  :       True,
            "image_size" :        224, # 299 for inceptionV3, 300 for efficientnet-b3, 224 for others
         'val_image_size':        224,
        "image_stats"    : ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] ),
                        
                        # imagenet_stat = ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
                        # PTCGA_stat = ( [0.7184, 0.5076, 0.6476], [0.0380, 0.0527, 0.0352] )
                        # unbiased_stat = ( [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] )
                        
            "seed"       :       1234,
            "epochs"     :        100,
            "checkpoint" :       None,
            "pretrained" :      False,
            "resume_fold":          0,
            "resume_epoch":         0, 
            "batch_size" :        128,  # 64 is ok for FP32  128 ok for mp @resnet50,   128 for inceptionV3 wAC, 64 for efficientnet
            "base_batch_size" :   512,
            "num_accum"  :          1,
            "n_fold"     :          3,
            "lr"         :       5e-4,  
         "optim"         :    'adamw',  # optimizer one of adam, adamw, sgd, lars
            "cold_epochs":          0,   
          "warmup_epochs":         10,  
         "snapshot_epoch":          1,
            "wd"         :     5.0e-5,  # set 0.0 for fine-tuning
            "clip_norm"  :        1.0,
            "dataset_dir": './datasets/PTCGA200',
            "root_dir"   : './',
            'nnode'      :   8,    # number of nodes
            'dist_url'   :   'tcp://127.0.0.1:52111', # master node:ephemeral_port 
}

if 'PTCGA200' in os.path.basename(config["dataset_dir"]):
    config['num_classes'] = 20
elif 'PCam' in os.path.basename(config["dataset_dir"]):
    config['num_classes'] = 2
elif 'PANDA' in os.path.basename(config["dataset_dir"]):
    config['num_classes'] = 6

config = easydict.EasyDict(config)

