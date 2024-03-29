import easydict
import os

config = {  
                'memo'   :  'resnet50_PCam200_scratch',
      'model_name'       :  'resnet50', # one of resnet18, resnet50, incv3, eff-b3, vit-s16, vit-b32
      'dropout_p'        :        0.4,
      'reset_bn'         :       True,
      'self_supervised' :       None, # one of None, byol. None indicates supervised training. BYOL only supports resnet now.  
      'fine_tune'        :      False, # When True base model is frozen and only heads are trained
      'enable_autocast'  :       True,
            "image_size" :        224, # 299 for inceptionV3, 300 for efficientnet-b3, 224 for others
         'val_image_size':        224,
        "image_stats"    : ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] ),
                        
                        # imagenet_stats = ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
                        # PTCGA_stats = ( [0.7184, 0.5076, 0.6476], [0.0380, 0.0527, 0.0352] )
                        # unbiased_stats = ( [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] )
                        
            "seed"       :       1234,
            "epochs"     :          5,
    "stop_num_updates"   :       1000,
            "checkpoint" :       None,
            "pretrained" :      False,
            "resume_epoch":         0, 
            "batch_size" :        128,  # batch size per GPU
            "num_workers":          1,  # num_workers parameter of DataLoader
            "base_batch_size" :   512,  # peak lr = (total_batch_size / base_batch_size) * lr
            "num_accum"  :          1,
            "n_fold"     :          3,
            "lr"         :       1e-4,  
         "optim"         :    'adamw',  # optimizer one of adam, adamw, sgd, lars
            "cold_epochs":          0,   
          "warmup_epochs":          1,  
         "snapshot_epoch":          1,
            "wd"         :     5.0e-5,  # set 0.0 for fine-tuning
            "clip_norm"  :        1.0,
            "dataset_dir": './datasets/PCam200',
            "root_dir"   : './',
            'nnode'      :   1,    # number of nodes
            'hosts'      :   [],   # list of hostnames, 
            'dist_url'   :   'tcp://127.0.0.1:52111', # master node:ephemeral_port 
}

if 'PTCGA200' in os.path.basename(config["dataset_dir"]):
    config['num_classes'] = 20
elif 'PCam' in os.path.basename(config["dataset_dir"]):
    config['num_classes'] = 2
elif 'PANDA' in os.path.basename(config["dataset_dir"]):
    config['num_classes'] = 6

config = easydict.EasyDict(config)

