import easydict

# original preset
_LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
_WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
_EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}


config = {  
            "debug"      :    False,
            "image_size" :     224, # 224 for default resnet50
            "seed"       :    3333,
            "n_fold"     :       3, 
            "epochs"     :     40,
            "checkpoint" :    None, 
            "batch_size" :      64,  # 128 out of memory wAC @16GB V100
            "base_batch_size" : 512,  # original 256; set total batch size to cancel warmup
            "num_accum"  :       1,
            "lr"         :     0.45,  
            "cold_epochs":        0,  
            "warmup_epochs":      10,  
            "wd"         :     1.0e-6,
            "momentum"   :      0.9,  # SGD momentum
      "trust_coefficient":     0.001,  # LARS coefficient 
            "clip_norm"  :     None,
        "synch_batchnorm":     True,
            "dataset_dir":    '/groups/gcc50585/migrated_from_SFA_GPFS/datasets/HE_200micron',
            "root_dir"   :    '/groups/gcc50585/migrated_from_SFA_GPFS/Github_repos/HE200_unsupervise/BYOL',
            "job_dir"    :    '/groups/gcc50585/migrated_from_SFA_GPFS/Github_repos/HE200_unsupervise',
      'enable_autocast'  :    True,
            
}


BYOL_config = {
          "num_classes"  :      20,
        "base_target_ema":    0.99,
 "projector_hidden_size" :    4096,
 "projector_output_size" :     256,
 "predictor_hidden_size" :    4096,
        "initializer"    :  "original",
}

config['model_config'] = BYOL_config 
config = easydict.EasyDict(config)